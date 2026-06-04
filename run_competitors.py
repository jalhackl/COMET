import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import tslearn

# import redpandda
# from redpandda import *
# import geostas

# import visualizations
# import comodo
import hdbscan
import time

# from redpandda import preprocessing, preprocess_protein_trajectory
import redpandda_general
from clustering_functions import clustering_workflow
from compare_clusterings import *
import clustering_functions
# from timestep_clustering import *
import distance_matrix as dm
import seaborn as sns
import compare_clusterings as cc

#### CAKMAK competitor ###

import logging
import json
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import silhouette_score, adjusted_mutual_info_score
from st_clustering_benchmark_modified import ST_DBSCAN, ST_Agglomerative, ST_KMeans, ST_OPTICS, ST_SpectralClustering, ST_AffinityPropagation, ST_BIRCH, ST_HDBSCAN
import threading




# control execution time of functions

TIMER = 120
PERMUT = 12

class TimeoutError(Exception):
    pass

class InterruptableThread(threading.Thread):
    def __init__(self, func, *args, **kwargs):
        threading.Thread.__init__(self)
        self._func = func
        self._args = args
        self._kwargs = kwargs
        self._result = None

    def run(self):
        self._result = self._func(*self._args, **self._kwargs)

    @property
    def result(self):
        return self._result


class timeout(object):
    def __init__(self, sec):
        self._sec = sec

    def __call__(self, f):
        def wrapped_f(*args, **kwargs):
            it = InterruptableThread(f, *args, **kwargs)
            it.start()
            it.join(self._sec)
            if not it.is_alive():
                return it.result
            raise TimeoutError('execution expired')
        return wrapped_f
    


def make_generator(parameters):
    """Helper function for st_grid_search. Returns a dictionary of all possible parameter combinations."""
    if not parameters:
        yield dict()
    else:
        key_to_iterate = list(parameters.keys())[0]
        next_round_parameters = {p : parameters[p]
                    for p in parameters if p != key_to_iterate}
        for val in parameters[key_to_iterate]:
            for pars in make_generator(next_round_parameters):
                temp_res = pars
                temp_res[key_to_iterate] = val
                yield temp_res
                
def st_silhouette_score(X, labels, eps1=0.05, eps2=10, metric='euclidean'):
    """Helper function for st_grid_search. Hyperparameter combinations are evaluated with the Silhouette score."""
    n, m = X.shape
    time_dist = pdist(X[:, 0].reshape(n, 1), metric=metric)
    euc_dist = pdist(X[:, 1:], metric=metric)

    # filter the euc_dist matrix using the time_dist
    dist = np.where(time_dist <= eps2, euc_dist, 2 * eps1)

    return silhouette_score(squareform(dist), labels, metric='precomputed')

@timeout(TIMER*PERMUT)
def st_grid_search(estimator, split, X, param_dict, metric, y=None, frame_size=None, frame_overlap=None):
    """
    Grid Search of hyperparameters for spatial-temporal clustering algorithms
    
    Parameters
    ----------
    estimator: class
        ST clustering algorithm
    split: boolean
        Flag to indicate whether whole X should be loaded in RAM or processed in smaller chunks.
    X: numpy array
        Data on which grid search is performed
    param_dict: dict
        Dictionary with parameters to be optimized as keys and value range of grid search as value.
    metric: str
        The metric to evaluate the clustering quality
    y: numpy array
        Optional. Some metrics compare predictions with ground truth. Then, labels need to be provided.
    frame_size: int
        Optional. If split is True, indicate how large the chunks should be.
    
    Returns
    -------
    param_opt
        Optimal hyperparameter combination
    """
    param_opt = None
    s_max = 0
    for param in make_generator(param_dict):
        clust = estimator(**param)
        if not split:
            clust.st_fit(X)
        else:
            clust.st_fit_frame_split(X, frame_size, frame_overlap)
            
        if param_opt is None: 
            param_opt = param
        
        # different performance evaluation metrics
        if metric=='silhouette':
            try:
                score = st_silhouette_score(X=X, labels=clust.labels, eps1=param['eps1'] , eps2=param['eps2'], metric='euclidean')
            except (TypeError, ValueError) as e:
                continue
            #print('Silhouette score for parameters {}: {}'.format(param,score))
        elif metric=='ami':
            score = adjusted_mutual_info_score(y,clust.labels)

        # store parameter combination if it outperforms given the metric
        if score > s_max:
            s_max = score
            param_opt = param
    return param_opt

@timeout(TIMER*PERMUT)
def traj_grid_search(estimator, X, param_dict, metric):
    """
    Grid Search of hyperparameters for spatial-temporal clustering algorithms
    
    Parameters
    ----------
    estimator: class
        ST clustering algorithm
    split: boolean
        Flag to indicate whether whole X should be loaded in RAM or processed in smaller chunks.
    X: numpy array
        Data on which grid search is performed
    param_dict: dict 
        Dictionary with parameters to be optimized as keys and value range of grid search as value.
    metric: str
        The metric to evaluate the clustering quality
    y: numpy array
        Optional. Some metrics compare predictions with ground truth. Then, labels need to be provided.
    frame_size: int
        Optional. If split is True, indicate how large the chunks should be.
    
    Returns
    -------
    param_opt
        Optimal hyperparameter combination
    """
    param_opt = {'detect_radius':40, 'similarity_threshold':0.5}
    s_max = 0
    for param in make_generator(param_dict):
        clust = estimator(**param)
        clust.st_fit(X)
        
        if param_opt is None: 
            param_opt = param
        
        # different performance evaluation metrics
        if metric=='silhouette':
            try:
                score = st_silhouette_score(X=X, labels=clust.labels, eps1=param['eps1'] , eps2=param['eps2'], metric='euclidean')
            except (TypeError, ValueError) as e:
                continue
            #print('Silhouette score for parameters {}: {}'.format(param,score))
        # elif metric=='ami':
        #     score = adjusted_mutual_info_score(clust.true_labels,clust.labels)
            #print('AMI score for parameters {}: {}'.format(param,score))
            
        # store parameter combination if it outperforms given the metric
        if score > s_max:
            s_max = score
            param_opt = param
    return param_opt


class Test(object):       
    # use this function with st clusterers
    @timeout(TIMER) # set seconds for timeout
    def frame_split_cluster(self, algorithm, data, frame_size, frame_overlap):
        import time
        start_time = time.time()
        algorithm.st_fit_frame_split(data, frame_size, frame_overlap)
        runtime = time.time() - start_time
        ami = adjusted_mutual_info_score(labels, algorithm.labels)
        return ami, runtime
        
    # use this with trajectory clustering
    @timeout(TIMER)
    def traj_cluster(self,algorithm, data):
        import time
        start_time = time.time()
        algorithm.st_fit(data)
        runtime = time.time() - start_time
        ami = adjusted_mutual_info_score(algorithm.true_labels, algorithm.labels)
        return ami, runtime
        
    # use this with dbscan2
    @timeout(TIMER)
    def cluster(self, algorithm, data):
        import time
        start_time = time.time()
        algorithm.st_fit(data)
        runtime = time.time() - start_time
        ami = adjusted_mutual_info_score(labels, algorithm.labels)
        return ami, runtime
    

def distance_matrix_grid_search(estimator, df, split, X, param_dict, metric, y=None, frame_size=None, frame_overlap=None):
    """
    Grid Search of hyperparameters for distance-based clustering algorithms
    
    Parameters
    ----------
    estimator: class
        ST clustering algorithm
    split: boolean
        Flag to indicate whether whole X should be loaded in RAM or processed in smaller chunks.
    X: numpy array
        Data on which grid search is performed
    param_dict: dict
        Dictionary with parameters to be optimized as keys and value range of grid search as value.
    metric: str
        The metric to evaluate the clustering quality
    y: numpy array
        Optional. Some metrics compare predictions with ground truth. Then, labels need to be provided.
    frame_size: int
        Optional. If split is True, indicate how large the chunks should be.
    
    Returns
    -------
    param_opt
        Optimal hyperparameter combination
    """
    param_opt = None
    s_max = 0
    for param in make_generator(param_dict):
        #clust = estimator(**param)
        #if not split:
        #    clust.st_fit(X)
        #else:
        #    clust.st_fit_frame_split(X, frame_size, frame_overlap)

        # in this case we are getting the results somewhat different


        #result = estimator(X, min_cluster_size=2, min_samples=2 ,return_matrix=False, stdev_addition=False)
        result = estimator(X, return_matrix=False, stdev_addition=False, **param)



        clustering =  list(result[0])

            
        if param_opt is None: 
            param_opt = param
        
        # different performance evaluation metrics
        if metric=='silhouette':
            try:
                score = st_silhouette_score(X=X, labels=clustering, eps1=param['eps1'] , eps2=param['eps2'], metric='euclidean')
            except (TypeError, ValueError) as e:
                continue
            #print('Silhouette score for parameters {}: {}'.format(param,score))
        elif metric=='ami':
            #score = adjusted_mutual_info_score(y,clustering)
            score = score_calc(df,clustering)

        # store parameter combination if it outperforms given the metric
        if score > s_max:
            s_max = score
            param_opt = param
    return param_opt

from sklearn.metrics import adjusted_mutual_info_score
import itertools
import copy
from clustering_functions import clustering_workflow

def clustering_workflow_grid_search_ami(traj_array, matrices_to_apply, base_clustering_algo, param_grid, y_true, post_process_noise=False, noise_label=-1):
    """
    Grid search for optimal clustering parameters using AMI as the scoring metric.

    Parameters
    ----------
    traj_array : numpy array
        The trajectory data.
    matrices_to_apply : list of str
        Types of matrices to generate (e.g., "delta", "stddv").
    base_clustering_algo : dict
        Dictionary with keys: 'name', 'method', and 'params'.
        'params' will be modified with values from param_grid.
    param_grid : dict
        Dictionary of hyperparameter names and lists of values to try.
    y_true : numpy array
        Ground truth labels for AMI evaluation.
    post_process_noise : bool
        Whether to assign noise points after clustering.
    noise_label : int
        Label used to denote noise in clustering.

    Returns
    -------
    best_params : dict
        The hyperparameters with the best AMI score.
    best_score : float
        The highest AMI score achieved.
    best_result : dict
        The result dict from clustering_workflow corresponding to the best score.
    """

    keys, values = zip(*param_grid.items())
    all_param_combos = [dict(zip(keys, v)) for v in itertools.product(*values)]

    best_score = float('-inf')
    best_params = None
    best_result = None

    for param_set in all_param_combos:
        clustering_algo = copy.deepcopy(base_clustering_algo)
        clustering_algo["params"].update(param_set)

        results = clustering_workflow(
            traj_array=traj_array,
            matrices_to_apply=matrices_to_apply,
            clusterings_to_apply=[clustering_algo],
            post_process_noise=post_process_noise,
            noise_label=noise_label,
            return_matrices=False
        )

        if not results:
            continue

        result = results[0]
        labels_pred = result["clustering"]

        try:
            score = adjusted_mutual_info_score(y_true, labels_pred)
        except Exception as e:
            print(f"Error with params {param_set}: {e}")
            continue

        if score > best_score:
            best_score = score
            best_params = param_set
            best_result = result

    return best_params, best_score, best_result

#substitutions = {'frame':'t', 'id':'obj_id','cid':'label','x':'x','y':'y'}
substitutions = {'t':'frame', 'obj_id':'id','label':'cid','x':'x','y':'y'}


def format_cluster_df(df, substitutions, add_z=True):
    filtered_df = df[list(substitutions.values())]

    # Step 2: Rename columns according to the dictionary keys
    filtered_df = filtered_df.rename(columns={v: k for k, v in substitutions.items()})

    if add_z:
        if 'z' not in df.columns:
            filtered_df['z'] = 0


    return filtered_df


def append_result(row):
    df_row = pd.DataFrame([row], columns=["dataset", "size", "algorithm", "runtime","n_objects"])
    df_row.to_csv(output_file, mode='a', header=not os.path.exists(output_file), index=False)

def optimize_clusterings_with_grid_search(traj_array, matrices_to_apply, clusterings_to_apply, param_grids, df, post_process_noise=False, noise_label=-1):
    """
    Perform grid search for each clustering algorithm in clusterings_to_apply for each matrix in matrices_to_apply, 
    optimizing only the parameters specified in param_grids, one matrix at a time.
    
    Parameters
    ----------
    traj_array : numpy array
        The trajectory data.
    matrices_to_apply : list of str
        The list of matrices to generate (e.g., "delta", "stddv").
    clusterings_to_apply : list of dicts
        List of dictionaries where each dict represents a clustering algorithm.
    param_grids : list of dicts
        List of parameter grids for each clustering algorithm to be optimized. If a grid is empty, no optimization will occur.
    df : df
        Ground truth labels for AMI evaluation.
    post_process_noise : bool
        Whether to assign noise points after clustering.
    noise_label : int
        Label used to denote noise.

    Returns
    -------
    optimized_parameters : dict
        A dictionary with matrix types as keys and another dictionary as values. This nested dictionary maps clustering algorithms to their optimized parameters for each matrix type.
    """
    optimized_parameters = {}


    # Loop over each matrix type in matrices_to_apply
    for matrix_type in matrices_to_apply:
        print(f"Optimizing clusterings for matrix type: {matrix_type}")
        
        optimized_parameters[matrix_type] = {}

        # Loop over each clustering algorithm
        for i, clustering_algo in enumerate(clusterings_to_apply):

            param_grid = param_grids[i]  # Get the parameter grid for this algorithm

            # If there are no parameters to optimize, skip the grid search
            if not param_grid:
                print(f"No parameters to optimize for {clustering_algo['name']} with matrix {matrix_type}, skipping grid search.")
                optimized_parameters[matrix_type][clustering_algo['name']] = clustering_algo["params"]
                continue

            # Run grid search for the current matrix-clustering combination
            best_params, best_score, best_result = clustering_workflow_grid_search_ami(
                traj_array=traj_array,
                matrices_to_apply=[matrix_type],  # Only this matrix type
                base_clustering_algo=clustering_algo,
                param_grid=param_grid,
                df=df,
                post_process_noise=post_process_noise,
                noise_label=noise_label
            )

            # Store the optimized parameters for the current matrix-clustering combination
            optimized_parameters[matrix_type][clustering_algo["name"]] = best_params


    return optimized_parameters

def update_clusterings_for_matrix(original_clusterings, optimized_params_dict, matrix_key):
                    updated_clusterings = []
                    for clustering in original_clusterings:
                        name = clustering["name"]
                        updated_params = optimized_params_dict[matrix_key].get(name, clustering["params"])
                        updated_clustering = {
                            "name": clustering["name"],
                            "method": clustering["method"],
                            "params": updated_params
                        }
                        updated_clusterings.append(updated_clustering)
                    return updated_clusterings


import os, random, re
import pandas as pd

data_folder = "/Users/work/Library/Mobile Documents/com~apple~CloudDocs/Desktop/ADesktop/Studium/PhD/DataMining/2025-COMET/COMET/full_data/"
with open("00results/runtime_analysis/dataset_selection_timepoints_balanced.txt", "r") as f:
    csv_files = [line.strip() for line in f if line.strip()]


# csv_files = csv_files_timepoints
output_file = "00results/runtime_analysis/cakmak_results_timepoints.csv"

# --- Settings ---
FRAME_SIZE = 10
FRAME_OVERLAP = 5
substitutions = {'t':'frame', 'obj_id':'id','label':'cid','x':'x','y':'y'}

st_runs = True

# --- Resume from existing results if needed ---
processed = set()
if os.path.exists(output_file):
    previous = pd.read_csv(output_file)
    processed = set(previous["Dataset"].unique())
    print(f" Resuming... already processed: {processed}")

# --- Helper for timed clustering ---
t = Test()  # from your previous helper functions

# --- Main Loop ---
for filename in csv_files:
    if filename in processed:
        print(f" Skipping {filename}, already processed.")
        continue

    print(f"\n Processing {filename} ...")
    df = pd.read_csv(os.path.join(data_folder, filename))

    # Prepare ST and trajectory data
    df_st = redpandda_general.mean_preprocess_dataframe_cakmak(df.copy())
    df = format_cluster_df(df, substitutions)
    df = redpandda_general.mean_preprocess_dataframe(df)
    traj_array, point_array, frames_count, n_objects = redpandda_general.prepare_data_from_df(df)

    # Normalize coordinates for ST clustering
    df_st['x'] = (df_st['x'] - df_st['x'].min()) / (df_st['x'].max() - df_st['x'].min())
    df_st['y'] = (df_st['y'] - df_st['y'].min()) / (df_st['y'].max() - df_st['y'].min())
    data = df_st[['frame','x','y']].values
    n_cluster = len(np.unique(df_st['cid'])) - (1 if -1 in df_st['cid'].values else 0)

    # --- ST clustering workflow (fixed parameters, no optimization) ---
    if st_runs:
    #     try:
    #         hdbscan = ST_HDBSCAN(eps2=25, min_cluster_size=max(2,n_cluster), min_samples=2)
    #         start = time.time()
    #         hdbscan.st_fit_frame_split(data, FRAME_SIZE, FRAME_OVERLAP)
    #         runtime = time.time() - start
    #         append_result([filename, frames_count, "ST_HDBSCAN", round(runtime,4), n_objects])
    #     except Exception as e:
    #         logging.error(f"HDBSCAN failed on {filename}: {e}")

        try:
            spectral = ST_SpectralClustering(n_clusters=n_cluster, eps2=15)
            start = time.time()
            spectral.st_fit_frame_split(data, FRAME_SIZE, FRAME_OVERLAP)
            runtime = time.time() - start
            append_result([filename, frames_count, "ST_Spectral", round(runtime,4), n_objects])
        except Exception as e:
            logging.error(f"Spectral failed on {filename}: {e}")
        

print("\n DONE! All files processed.")

