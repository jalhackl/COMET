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

data_folder = "/COMET/full_data/"
output_file = "couzin_results.csv"
# output_file = "reynolds_results.csv"

selection_size = 10

# List all CSV files that match your pattern
all_csv_files = [f for f in os.listdir(data_folder) if f.startswith("couzin_") and f.endswith(".csv")]

# Pick random files
random.seed(42)
selected_csv_files = random.sample(all_csv_files, selection_size)
csv_files = selected_csv_files

# Frame parameters
FRAME_SIZE = 10
FRAME_OVERLAP = 5

# SHiP configuration
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

# Column mapping
substitutions = {'t':'frame', 'obj_id':'id','label':'cid','x':'x','y':'y'}

# ---------------- Helper functions ----------------
def append_result(row):
    df_row = pd.DataFrame([row], columns=["dataset", "size", "algorithm", "runtime","n_objects"])
    df_row.to_csv(output_file, mode='a', header=not os.path.exists(output_file), index=False)

def format_cluster_df(df, substitutions, add_z=True):
    filtered_df = df[list(substitutions.values())]
    filtered_df = filtered_df.rename(columns={v: k for k, v in substitutions.items()})
    if add_z and 'z' not in df.columns:
        filtered_df['z'] = 0
    return filtered_df

# Resume from existing results
processed = set()
if os.path.exists(output_file):
    previous = pd.read_csv(output_file)
    processed = set(previous["dataset"].unique())
    print(f"✅ Resuming... already processed: {processed}")

# ---------------- Main Loop ----------------
if __name__ == '__main__':
    for filename in csv_files:
        if filename in processed:
            print(f"⏭ Skipping {filename}, already processed.")
            continue

        print(f"\n🚀 Processing {filename} ...")
        df = pd.read_csv(os.path.join(data_folder, filename))
        df = format_cluster_df(df, substitutions)

        # Prepare trajectory data
        traj_array, point_array, frames_count, n_objects = redpandda_general.prepare_data_from_df(df)

        # ----- SHiP -----
        print("⚙️ Running SHiP...")
        start = time.time()
        res = clustering_workflow(traj_array, matrices_to_apply, clusterings_to_apply, post_process_noise=True)
        runtime = time.time() - start
        append_result([filename, frames_count, "COMET", round(runtime, 4),n_objects])
        print("⚙️ SHip completed")
        
        
        # ----- ST clustering -----
        df = pd.read_csv(os.path.join(data_folder, filename))
        df_st = df.copy()
        df_st = redpandda_general.mean_preprocess_dataframe_cakmak(df_st)
        df = format_cluster_df(df, substitutions)
        datapoints_original =  len(df)
        df = redpandda_general.mean_preprocess_dataframe(df)
        trajectories = df
        grouped = df.groupby('obj_id')
        dataset_size = frames_count

        # --- Prepare labels + normalized input for ST competitor ---
        df_st['x'] = (df_st['x'] - df_st['x'].min()) / (df_st['x'].max() - df_st['x'].min())
        df_st['y'] = (df_st['y'] - df_st['y'].min()) / (df_st['y'].max() - df_st['y'].min())

        data = df_st[['frame', 'x', 'y']].values
        labels = df_st['cid'].to_numpy()

        n_cluster = len(np.unique(labels)) - (1 if -1 in labels else 0)

        from st_clustering_benchmark_modified import ST_HDBSCAN, ST_SpectralClustering

        class Test:
            def frame_split_cluster(self, algorithm, data, frame_size, frame_overlap):
                start_time = time.time()
                algorithm.st_fit_frame_split(data, frame_size, frame_overlap)
                runtime = time.time() - start_time
                return adjusted_mutual_info_score(labels, algorithm.labels), runtime
        print("⚙️ Preparing ST clustering...")
        df_st['x'] = (df_st['x'] - df_st['x'].min()) / (df_st['x'].max() - df_st['x'].min())
        df_st['y'] = (df_st['y'] - df_st['y'].min()) / (df_st['y'].max() - df_st['y'].min())

        print(df_st.columns)
        data = df_st[['frame', 'x', 'y']].values
        labels = df_st['cid'].to_numpy()
        n_cluster = len(np.unique(labels)) - (1 if -1 in labels else 0)

        t = Test()

        # ST-HDBSCAN
        print("⚙️ Running ST_HDBSCAN...")
        hdbscan = ST_HDBSCAN(eps2=25, min_cluster_size=n_cluster, min_samples=2)
        ami_score, runtime = t.frame_split_cluster(hdbscan, data, FRAME_SIZE, FRAME_OVERLAP)
        print(f"✅ ST_HDBSCAN done: AMI={ami_score}, runtime={runtime:.2f}s")
        append_result([filename, frames_count, "ST_HDBSCAN", round(runtime, 4),n_objects])

        # # ST-Spectral
        # print("⚙️ Running ST_Spectral...")
        # spectral = ST_SpectralClustering(eps2=15, n_clusters=n_cluster)
        # ami_score, runtime = t.frame_split_cluster(spectral, data, FRAME_SIZE, FRAME_OVERLAP)
        # print(f"✅ ST_Spectral done: AMI={ami_score}, runtime={runtime:.2f}s")
        # append_result([filename, frames_count, "ST_Spectral", round(runtime, 4),n_objects])


print("\n✅ DONE! Results saved to couzin_results.csv")
