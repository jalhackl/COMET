import os
import time
import signal
import logging
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

import redpandda_general
from st_clustering_benchmark_modified import (
    ST_DBSCAN,
    ST_SpectralClustering,
    ST_HDBSCAN
)

# ============================================================
# CONFIGURATION
# ============================================================

DATA_FOLDER = "/COMET/full_data/"
DATASET_LIST = "00results/runtime_analysis/dataset_selection_timepoints_balanced.txt"
OUTPUT_FILE = "00results/runtime_analysis/cakmak_competitors.csv"

FRAME_SIZE = 10
FRAME_OVERLAP = 5
MAX_RUNTIME = 200  # seconds

# ============================================================
# TIMEOUT HANDLING (Unix only, same as tslearn script)
# ============================================================

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException()

# ============================================================
# HELPERS
# ============================================================

def object_level_labels(df_points, point_labels):
    """
    Majority vote from point-level labels to object-level labels.
    """
    df_tmp = df_points.copy()
    df_tmp["pred"] = point_labels

    obj_labels = {}
    for obj_id, g in df_tmp.groupby("id"):
        labels = g["pred"].values
        labels = labels[labels != -1]  # ignore noise
        if len(labels) == 0:
            obj_labels[obj_id] = -1
        else:
            obj_labels[obj_id] = Counter(labels).most_common(1)[0][0]

    return obj_labels


def save_now(results):
    df = pd.DataFrame(
        results,
        columns=[
            "Dataset", "Timepoints", "Objects",
            "Algorithm", "Status",
            "Runtime", "ARI", "NMI"
        ]
    )
    df.to_csv(OUTPUT_FILE, index=False)
    print(" Saved progress.")

# ============================================================
# LOAD DATASETS
# ============================================================

with open(DATASET_LIST, "r") as f:
    csv_files = [line.strip() for line in f if line.strip()]
    csv_files.remove("reynolds_900_51.csv")  # known problematic dataset

processed = set()
results = []

if os.path.exists(OUTPUT_FILE):
    prev = pd.read_csv(OUTPUT_FILE)
    processed = set(prev["Dataset"].unique())
    results = prev.values.tolist()
    print(f"Resuming… already processed {len(processed)} datasets.")

# ============================================================
# MAIN LOOP
# ============================================================

for filename in csv_files[:5]:

    if filename in processed:
        print(f" Skipping {filename}, already processed.")
        continue

    print(f"\n Processing {filename}")
    df_raw = pd.read_csv(os.path.join(DATA_FOLDER, filename))
    try:
        # ----------------------------------------------------
        # Load + preprocess
        # ----------------------------------------------------
        df_raw = pd.read_csv(os.path.join(DATA_FOLDER, filename))

        # safety: ensure expected columns exist
        required_cols = {"frame", "id", "x", "y", "cid"}
        missing = required_cols - set(df_raw.columns)
        if missing:
            raise ValueError(f"Missing columns {missing} in {filename}")


        df = redpandda_general.mean_preprocess_dataframe(df_raw.copy())
        
        print(df.head())
        
        traj_array, point_array, frames_count, n_objects = \
            redpandda_general.prepare_data_from_df(df)

        # ST preprocessing (point-level)
        df_st = redpandda_general.mean_preprocess_dataframe_cakmak(df_raw.copy())

        # trajectory preprocessing (object-level)
        df_traj = redpandda_general.mean_preprocess_dataframe(df_raw.copy())
        traj_array, point_array, frames_count, n_objects = \
            redpandda_general.prepare_data_from_df(df_traj)

        # normalize spatial coords
        # normalize spatial coords for ST clustering
        df_st["x"] = (df_st["x"] - df_st["x"].min()) / (df_st["x"].max() - df_st["x"].min())
        df_st["y"] = (df_st["y"] - df_st["y"].min()) / (df_st["y"].max() - df_st["y"].min())

        X = df_st[["frame", "x", "y"]].values

        # ground truth (object-level)
        true_obj_labels = (
            df_st.groupby("id")["cid"]
            .first()
            .to_dict()
        )

        n_clusters = len(set(true_obj_labels.values())) - (
            1 if -1 in true_obj_labels.values() else 0
        )

        # ----------------------------------------------------
        # Algorithms
        # ----------------------------------------------------
        print(df_st.head())
        print(df_st.columns)

        algorithms = [
            ("ST_Spectral",
             ST_SpectralClustering(n_clusters=n_clusters, eps2=15)),

            ("ST_DBSCAN",
             ST_DBSCAN(eps1=0.05, eps2=15, min_samples=5)),

            ("ST_HDBSCAN",
             ST_HDBSCAN(eps2=15, min_cluster_size=max(2, n_clusters)))
        ]

        for name, algo in algorithms:
            print(f"  → {name}")

            try:
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(MAX_RUNTIME)

                start = time.perf_counter()
                algo.st_fit_frame_split(X, FRAME_SIZE, FRAME_OVERLAP)
                runtime = time.perf_counter() - start

                signal.alarm(0)

                # object-level projection
                pred_obj_labels = object_level_labels(df_st, algo.labels)

                y_true = []
                y_pred = []
                for obj_id in true_obj_labels:
                    y_true.append(true_obj_labels[obj_id])
                    y_pred.append(pred_obj_labels.get(obj_id, -1))

                ari = adjusted_rand_score(y_true, y_pred)
                nmi = normalized_mutual_info_score(y_true, y_pred)

                results.append([
                    filename,
                    frames_count,
                    n_objects,
                    name,
                    "ok",
                    round(runtime, 4),
                    round(ari, 4),
                    round(nmi, 4)
                ])

                print(f"    runtime={runtime:.2f}s | ARI={ari:.3f} | NMI={nmi:.3f}")
                save_now(results)

            except TimeoutException:
                signal.alarm(0)
                logging.warning(f"{name} timed out on {filename}")
                print(f"    {name} skipped (>{MAX_RUNTIME}s)")

                results.append([
                    filename,
                    frames_count,
                    n_objects,
                    name,
                    "timeout",
                    MAX_RUNTIME,
                    np.nan,
                    np.nan
                ])
                save_now(results)

            except Exception as e:
                signal.alarm(0)
                logging.error(f"{name} failed on {filename}: {e}")

                results.append([
                    filename,
                    frames_count,
                    n_objects,
                    name,
                    "failed",
                    np.nan,
                    np.nan,
                    np.nan
                ])
                save_now(results)

    except Exception as e:
        logging.error(f"Dataset-level failure on {filename}: {e}")

print("\n Finished all datasets.")
print("Results stored in:", OUTPUT_FILE)
