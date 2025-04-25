import numpy as np
import distance_matrix as dm
import redpandda_general
from postprocess_clusterings import assign_noise_points

def clustering_workflow(traj_array, matrices_to_apply, clusterings_to_apply, post_process_noise = False, noise_label = -1, return_matrices = False):

    dist_matrices = redpandda_general.get_distance_matrices(traj_array)
    average_distance_matrix = redpandda_general.calculate_average_delta_matrix(dist_matrices)
    std_distance_matrix = redpandda_general.get_std_matrices(dist_matrices)

    matrices_for_computations = {}
    import time
    times_matrices =  {}


    if "delta" in matrices_to_apply:
        
        start_time = time.time()
        delta_matrices = redpandda_general.get_delta_matrices(dist_matrices)
        average_delta_matrix = redpandda_general.calculate_average_delta_matrix(delta_matrices)
        std_delta_matrix = redpandda_general.get_std_matrices(delta_matrices)
        matrices_for_computations["delta"] = average_delta_matrix

        curr_time = time.time() - start_time 
        times_matrices["delta"] = curr_time

    if "delta w/o abs" in matrices_to_apply:
        start_time = time.time()

        delta_matrices_wo_absolute = redpandda_general.get_delta_matrices_wo_absolute(dist_matrices)
        average_delta_matrix_wo_absolute = redpandda_general.calculate_average_delta_matrix(delta_matrices_wo_absolute)
        matrices_for_computations["delta w/o abs"] = average_delta_matrix_wo_absolute

        curr_time = time.time() - start_time 
        times_matrices["delta w/o abs"] = curr_time

    if "stddv" in matrices_to_apply:
        start_time = time.time()

        stddv_matrices = redpandda_general.get_stddv(dist_matrices)
        matrices_for_computations["stddv"] = stddv_matrices

        curr_time = time.time() - start_time 
        times_matrices["stddv"] = curr_time



    if "delta+1std" in matrices_to_apply:
        start_time = time.time()


        summed_delta_matrix_1std = average_delta_matrix + std_delta_matrix 
        matrices_for_computations["delta+1std"] = summed_delta_matrix_1std

        curr_time = time.time() - start_time 
        times_matrices["delta+1std"] = curr_time + times_matrices["delta"]

    if "delta+2std" in matrices_to_apply:
        start_time = time.time()

        summed_delta_matrix_2std = average_delta_matrix + std_delta_matrix * 2 
        matrices_for_computations["delta+2std"] = summed_delta_matrix_2std

        curr_time = time.time() - start_time 
        times_matrices["delta+2std"] = curr_time + times_matrices["delta"]

    clustering_results = []
    for matrix in matrices_for_computations:

        for clustering_algo in clusterings_to_apply:
            start_time = time.time()
            new_clustering_results = dm.clustering_on_deltas(matrices_for_computations[matrix], clustering_algo["method"], **clustering_algo["params"])
            if isinstance(new_clustering_results, tuple):
                new_clustering_results = new_clustering_results[0]
            clustering_time = time.time() - start_time 
            total_time = clustering_time + times_matrices[matrix]

            if post_process_noise and noise_label in new_clustering_results:
                assign_noise_points(matrices_for_computations[matrix], new_clustering_results, noise_label)

            #clustering_results.append([clustering_algo["name"], clustering_algo["method"], clustering_algo["params"], matrix, new_clustering_results ])
            clustering_results.append({"name":clustering_algo["name"], "method":clustering_algo["method"], "params":clustering_algo["params"], "matrix":matrix, "clustering":new_clustering_results, "runtime": total_time })


    if return_matrices:
        matrices_for_computations["all distances"] = dist_matrices
        if "delta" in matrices_to_apply:
            matrices_for_computations["all deltas"] = delta_matrices
        if "delta w/o abs" in matrices_to_apply:
            matrices_for_computations["delta w/o abs"] = delta_matrices_wo_absolute

        return clustering_results, matrices_for_computations
    else:
        return clustering_results



def cluster_timesteps_change_points(input_matrix, change_points, clustering_method = "hdbscan", clustering_params ={}):
    start_idx = 0
    timestep_clusterings = []
    if len(change_points) < 1:
        change_points = [len(input_matrix) -1]
    for end_idx in change_points:
        timestep_clustering = {}
        timestep_clustering["start"] = start_idx
        timestep_clustering["end"] = end_idx

        delta_selected_frames = input_matrix[start_idx:end_idx + 1]  # Select range from start to end (inclusive)
        
        start_idx = end_idx + 1  # Move to the next range
        average_delta_matrix = redpandda_general.calculate_average_delta_matrix(delta_selected_frames)

        new_clustering_results = dm.clustering_on_deltas(average_delta_matrix, clustering_method, **clustering_params)
        timestep_clustering["clustering"] = new_clustering_results
        timestep_clusterings.append(timestep_clustering)

    # last cluster timeframe
    timestep_clustering = {}
    timestep_clustering["start"] = end_idx
    timestep_clustering["end"] = len(input_matrix)-1
    if timestep_clustering["end"] - timestep_clustering["start"] > 0:
        delta_selected_frames = input_matrix[end_idx:len(input_matrix)-1]  # Select range from start to end (inclusive)

        average_delta_matrix = redpandda_general.calculate_average_delta_matrix(delta_selected_frames)

        new_clustering_results = dm.clustering_on_deltas(average_delta_matrix, clustering_method, **clustering_params)
        timestep_clustering["clustering"] = new_clustering_results
        timestep_clusterings.append(timestep_clustering)
    return timestep_clusterings


def cluster_timesteps_from_timestep_clustering(input_matrix, time_labels, clustering_method = "hdbscan", clustering_params ={}):
    timestep_clusterings = []

    unique_labels = np.unique(time_labels)

    for label in unique_labels:
        timestep_clustering = {}

        indices = np.where(time_labels == label)[0]  # Find indices where label appears
        delta_selected_frames = input_matrix[indices]  # Select corresponding frames

        timestep_clustering["indices"] = indices

        average_delta_matrix = redpandda_general.calculate_average_delta_matrix(delta_selected_frames)

        new_clustering_results = dm.clustering_on_deltas(average_delta_matrix, clustering_method, **clustering_params)
        timestep_clustering["clustering"] = new_clustering_results
        timestep_clusterings.append(timestep_clustering)
    return timestep_clusterings

