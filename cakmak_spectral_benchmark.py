import os
import time
import random
import pandas as pd
import numpy as np
from sklearn.metrics import adjusted_mutual_info_score
import signal
import pandas as pd
import numpy as np
import os
import time
import random

from sklearn.metrics import adjusted_mutual_info_score
from clustering_functions import clustering_workflow
import redpandda_general
from SHiP import SHiP
from SHiP.partitioning import PartitioningMethod as PMethod
# =============================================================================
# Timeout helper
# =============================================================================
class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException()

def run_with_timeout(func, args=(), kwargs={}, timeout=120, fallback=20):
    """Runs a function with Timeout. If timeout reached -> return fallback runtime."""
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout)
    try:
        result = func(*args, **kwargs)
        signal.alarm(0)
        return result
    except TimeoutException:
        print(f"⏳ Timeout reached in {func.__name__}, assigning runtime={fallback}s")
        return None, fallback
    except Exception as e:
        print(f"⚠️ Error in {func.__name__}: {e}")
        return None, fallback

# =============================================================================
# Benchmark setup
# =============================================================================
data_folder = "COMET/full_data/"
output_file = "calovi_results_spectral.csv"
selection_size = 100

all_csv_files = [f for f in os.listdir(data_folder) if f.startswith("calovi_") and f.endswith(".csv")]
random.seed(42)
csv_files = random.sample(all_csv_files, selection_size)

FRAME_SIZE = 10
FRAME_OVERLAP = 5

matrices_to_apply = ["delta+1std"]
clusterings_to_apply = [{
    "name": "SHiP_1",
    "method": "ship",
    "params": {
        "partitioning_method": PMethod.Elbow,
        "hierarchie": 2,
        "tiebreaker_method": "euclidean_distance"
    }
}]

substitutions = {'t': 'frame', 'obj_id': 'id', 'label': 'cid', 'x': 'x', 'y': 'y'}

def append_result(row):
    df_row = pd.DataFrame([row], columns=["dataset", "size", "algorithm", "runtime", "n_objects"])
    df_row.to_csv(output_file, mode='a', header=not os.path.exists(output_file), index=False)

def format_cluster_df(df, substitutions, add_z=True):
    filtered = df[list(substitutions.values())]
    filtered = filtered.rename(columns={v: k for k, v in substitutions.items()})
    if add_z and 'z' not in df.columns:
        filtered['z'] = 0
    return filtered

# Resume previously processed
processed = set()
if os.path.exists(output_file):
    previous = pd.read_csv(output_file)
    processed = set(previous["dataset"].unique())
    print(f"✅ Resuming... already processed: {processed}")

# =============================================================================
# Main Loop
# =============================================================================
if __name__ == '__main__':
    for filename in csv_files:
        if filename in processed:
            print(f"⏭ Skipping {filename}, already processed.")
            continue

        print(f"\n🚀 Processing {filename} ...")
        df = pd.read_csv(os.path.join(data_folder, filename))
        df = format_cluster_df(df, substitutions)

        traj_array, point_array, frames_count, n_objects = redpandda_general.prepare_data_from_df(df)

        # # === COMET / SHiP RUN ===
        # print("⚙️ Running COMET (SHiP)...")
        # start = time.time()
        # res = clustering_workflow(traj_array, matrices_to_apply, clusterings_to_apply, post_process_noise=True)
        # runtime = time.time() - start
        # append_result([filename, frames_count, "COMET", round(runtime, 4), n_objects])
        # print(f"✅ COMET completed in {runtime:.2f}s")

        # === ST competitor preparation ===
        df_raw = pd.read_csv(os.path.join(data_folder, filename))
        df_st = redpandda_general.mean_preprocess_dataframe_cakmak(df_raw)
        df_prep = format_cluster_df(df_raw, substitutions)
        df_prep = redpandda_general.mean_preprocess_dataframe(df_prep)

        df_st['x'] = (df_st['x'] - df_st['x'].min()) / (df_st['x'].max() - df_st['x'].min())
        df_st['y'] = (df_st['y'] - df_st['y'].min()) / (df_st['y'] - df_st['y'].min()).max()

        data = df_st[['frame', 'x', 'y']].values
        labels = df_st['cid'].to_numpy()
        n_cluster = len(np.unique(labels)) - (1 if -1 in labels else 0)

        class Test:
            def frame_split_cluster(self, algorithm, data, frame_size, frame_overlap):
                start_time = time.time()
                algorithm.st_fit_frame_split(data, frame_size, frame_overlap)
                runtime = time.time() - start_time
                return adjusted_mutual_info_score(labels, algorithm.labels), runtime

        t = Test()

        # === ST-HDBSCAN ===
        # print("⚙️ Running ST_HDBSCAN (max 2 min)...")
        # hdbscan = ST_HDBSCAN(eps2=25, min_cluster_size=n_cluster, min_samples=2)
        # ami, runtime = run_with_timeout(t.frame_split_cluster,
        #                                 args=(hdbscan, data, FRAME_SIZE, FRAME_OVERLAP),
        #                                 timeout=120, fallback=20)
        # append_result([filename, frames_count, "ST_HDBSCAN", round(runtime, 4), n_objects])

        # === ST-Spectral ===
        print("⚙️ Running ST_Spectral (max 2 min)...")
        spectral = ST_SpectralClustering(eps2=15, n_clusters=n_cluster)
        ami, runtime = run_with_timeout(t.frame_split_cluster,
                                        args=(spectral, data, FRAME_SIZE, FRAME_OVERLAP),
                                        timeout=120, fallback=120)
        append_result([filename, frames_count, "ST_Spectral", round(runtime, 4), n_objects])

    print("\n✅ DONE! Results saved to", output_file)
