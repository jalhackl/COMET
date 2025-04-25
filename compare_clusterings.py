import numpy as np
import mdtraj as md
import CONSTANTS
import sklearn.metrics as m
from itertools import combinations
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix
import sklearn
from scipy.spatial import procrustes
from scipy.spatial.transform import Rotation as R
from joblib import Parallel, delayed

def geostas_csv_to_numpy_array(filename,k) -> np.ndarray:
  '''
  filename: Name of CSV-file in geostas folder (w/o path) including file format
  k: Cluster count
  Returns the geostas clustering as numpy array
  '''
  geostas_domains = np.genfromtxt(CONSTANTS.geostas_prefix + filename, delimiter = ',')
  geostas_domains = geostas_domains[1:len(geostas_domains)]
  geostas_domains = np.array([x[1] for x in geostas_domains], dtype=int)
  geostas_domains[geostas_domains == k] = 0
  return geostas_domains

def ca_geostas_clustering(geostas_clustering, traj):
  clustering = np.zeros(traj.n_residues)
  for atom in traj.topology.atoms:
    if atom.name == 'CA':
      clustering[atom.residue.index] = geostas_clustering[atom.index]
  return clustering


def get_corr_kiss_rmse(kiss_arr, rmse_arr):
  rmse_arr = rmse_arr[:len(kiss_arr)]
  delta_rmse = np.absolute(np.diff(rmse_arr,axis=0))
  return np.corrcoef(kiss_arr[:-1],delta_rmse)[1,0]

def get_AMI_of_clusterings(clustering1, clustering2):
  return m.adjusted_mutual_info_score(clustering1, clustering2)

def get_RMSE_for_summed_version(folded_dist: np.ndarray, current_dists: np.ndarray):
  '''
  folded_dist: Matrix containing distance matrix of protein in folded form
  current_dists: List of distance matrices of the current frames looked at
  
  Returns the RMSE (mean of the RMSEs of each of the summed frames)
  '''
  all_rmses = np.sqrt(np.sum(np.sum(np.square(np.subtract(folded_dist,current_dists)),axis=1),axis=1))
  return np.sum(all_rmses) / len(all_rmses)

def get_RMSE_of_clusters_and_total_for_lags(folded_dist: np.ndarray, current_dists: np.ndarray, lag_times: int, clustering: np.ndarray, k: int):
  '''
  folded_dist: Matrix containing distance matrix of protein in folded form
  current_dists: List of distance matrices of the current frames looked at
  lag-times: Amount of frames getting summed up
  clustering: Array containing index of cluster for each atom
  k: Amount of clusters
  
  Returns the RMSE (mean of the RMSEs of each frame) in total, for each clusters and the mean RMSE of the clusters
  '''
  squared_diffs = np.square(np.subtract(folded_dist,current_dists))
  total_rmses_for_frames = np.sqrt(np.sum(np.sum(squared_diffs,axis=1),axis=1))
  total_rmse = np.sum(total_rmses_for_frames) / lag_times

  cluster_rmses = np.zeros(k, dtype=float)
  
  for i in range(k):
    indices = np.where(clustering == i)[0]
    cluster_diffs = squared_diffs[np.ix_(range(lag_times), indices, indices)]
    cluster_rmses[i] = np.sum(np.sqrt(np.sum(np.sum(cluster_diffs,axis=1),axis=1))) / lag_times

  clusters_mean_rmse = np.sum(cluster_rmses) / k

  return total_rmse, cluster_rmses, clusters_mean_rmse

def get_RMSE_of_clusters_and_total(folded_dist: np.ndarray, current_dists: np.ndarray, clustering: np.ndarray, k: int):
  '''
  folded_dist: Matrix containing distance matrix of protein in folded form
  current_dists: List of distance matrices
  clustering: Array containing index of cluster for each atom
  k: Amount of clusters
  
  Returns an array of RMSEs of each frame, 
  an array |clusters|x|frames| containing arrays of the RMSEs of each cluster for each frame and
  an array containing the mean RMSE of the clusters for each frame
  '''
  ca_count = len(folded_dist)
  squared_diffs = np.square(np.subtract(folded_dist,current_dists))
  total_rmses_for_frames = np.sqrt((np.sum(np.sum(squared_diffs,axis=1),axis=1))*(1/ca_count))

  cluster_rmses = np.zeros(k, dtype=np.ndarray)
  new_k = k
  for i in range(k):
    indices = np.where(clustering == i)[0]
    if len(indices) <= 1:
      cluster_rmses[i] = np.zeros(len(current_dists))
      print('Empty cluster found')
      new_k -= 1
      continue
    ca_count = len(indices)
    cluster_diffs = squared_diffs[np.ix_(range(len(current_dists)), indices, indices)]
    cluster_rmses[i] = np.sqrt((np.sum(np.sum(cluster_diffs,axis=1),axis=1))*(1/ca_count))

  clusters_mean_rmses = np.sum(cluster_rmses, axis=0) / new_k

  return total_rmses_for_frames, cluster_rmses, clusters_mean_rmses

def get_RMSE_for_clustering(folded_dist: np.ndarray, current_dists: np.ndarray, clustering: np.ndarray, k: int):
  '''
  folded_dist: Matrix containing distance matrix of protein in folded form
  current_dists: List of distance matrices
  clustering: Array containing index of cluster for each atom
  k: Amount of clusters
  
  Returns an array of the mean RMSEs of the clusters for each frame
  '''
  squared_diffs = np.square(np.subtract(folded_dist,current_dists))
  cluster_rmses = np.zeros(k, dtype=np.ndarray)
  new_k = k
  for i in range(k):
    indices = np.where(clustering == i)[0]
    if len(indices) <= 1:
      cluster_rmses[i] = np.zeros(len(current_dists))
      print('Empty cluster found')
      new_k -= 1
      continue
    ca_count = len(indices)
    cluster_diffs = squared_diffs[np.ix_(range(len(current_dists)), indices, indices)]
    cluster_rmses[i] = np.sqrt((np.sum(np.sum(cluster_diffs,axis=1),axis=1))*(1/ca_count))
  clusters_mean_rmses = np.sum(cluster_rmses, axis=0) / new_k
  return clusters_mean_rmses

def get_Q_wo_clustering(current_dists: np.ndarray, use_max = True):
  '''
  current_dists: List of distance matrices

  Returns the Q without the clusters
  '''
  current_dists = current_dists[0:len(current_dists):int(len(current_dists)/400)] # TODO: Nur drin, um Rechnung kürzer zu machen
  combis = np.array(list(combinations(current_dists, 2)))
  rmse_for_frame_combis = np.sqrt(np.sum(np.square(np.array([x[0] - x[1] for x in combis]).reshape(len(combis),-1)),axis=1))
  if use_max:
    return np.max(rmse_for_frame_combis)
  else:
    return np.mean(rmse_for_frame_combis)
   

def get_Q_for_clustering_wo_parallel(current_dists: np.ndarray, clustering: np.ndarray, k: int, use_max = False):
  '''
  current_dists: List of distance matrices
  clustering: Array containing index of cluster for each atom
  k: Amount of clusters

  Returns the Q and an array with the max RMSE of each cluster
  '''
  cluster_qs = np.zeros(k, dtype=np.ndarray)
  current_dists = current_dists[0:len(current_dists):int(len(current_dists)/400)] # TODO: Nur drin, um Rechnung kürzer zu machen

  for i in range(k):
    indices = np.where(clustering == i)[0]
    ca_count = len(indices)
    if len(indices) <= 1:
      cluster_qs[i] = 0
      continue
    cluster_dists = current_dists[np.ix_(range(len(current_dists)), indices, indices)]
    combis = np.array(list(combinations(cluster_dists, 2)))
    rmse_for_frame_combis = np.sqrt((np.sum(np.square(np.array([x[0] - x[1] for x in combis]).reshape(len(combis),-1)),axis=1))*(1/ca_count))
    
    if not use_max:
      cluster_qs[i] = np.mean(rmse_for_frame_combis[rmse_for_frame_combis != 0]) # ResiCon uses max
    else:
      cluster_qs[i] = np.max(rmse_for_frame_combis[rmse_for_frame_combis != 0]) # ResiCon uses max


  return np.mean(cluster_qs), cluster_qs


from joblib import Parallel, delayed
import numpy as np
from itertools import combinations

def get_Q_for_clustering_parallel(current_dists: np.ndarray, clustering: np.ndarray, k: int, use_max=False, n_jobs=-1):
    cluster_qs = np.zeros(k, dtype=float)

    def compute_cluster_Q(i):
        indices = np.where(clustering == i)[0]
        ca_count = len(indices)
        if ca_count <= 1:
            return 0  # If there's only 1 element, return 0

        cluster_dists = current_dists[np.ix_(range(len(current_dists)), indices, indices)]
        combis = combinations(cluster_dists, 2)  # Lazy iteration instead of storing all at once

        rmse_values = []
        for x in combis:
            rmse = np.sqrt(np.sum((x[0] - x[1])**2) / ca_count)
            if rmse > 0:
                rmse_values.append(rmse)

        return np.max(rmse_values) if use_max else np.mean(rmse_values)

    # Run computations in parallel across clusters
    cluster_qs = np.array(Parallel(n_jobs=n_jobs)(delayed(compute_cluster_Q)(i) for i in range(k)))

    return np.mean(cluster_qs), cluster_qs



def compute_rmse_for_cluster(cluster_dists, ca_count, use_max):
    """
    Helper function to compute the RMSE for a given cluster.
    This function is parallelized across clusters.
    """
    if ca_count <= 1:
        return 0  # If there's only 1 element, return 0 (no meaningful RMSE)

    rmse_values = []
    for frame_1, frame_2 in combinations(cluster_dists, 2):  # Memory-efficient pairwise selection
        rmse = np.sqrt(np.sum((frame_1 - frame_2) ** 2) / ca_count)
        if rmse > 0:  # Ignore zero RMSE values
            rmse_values.append(rmse)

    if not rmse_values:
        return 0  # If no RMSE values remain, return 0

    return np.max(rmse_values) if use_max else np.mean(rmse_values)


def get_Q_for_clustering(current_dists: np.ndarray, clustering: np.ndarray, k: int, num_frames: int = 400, use_max=False, n_jobs=-1):
    """
    Computes the clustering quality (Q) based on RMSE values.

    Parameters:
    - current_dists: np.ndarray (shape: [frames, atoms, atoms]), list of distance matrices
    - clustering: np.ndarray, array containing index of cluster for each atom
    - k: int, number of clusters
    - num_frames: int, number of frames to use for computation (default=400)
    - use_max: bool, if True, use max RMSE; otherwise, use mean RMSE
    - n_jobs: int, number of parallel jobs (-1 uses all available cores)

    Returns:
    - mean_Q (float): Mean Q value across clusters
    - cluster_qs (np.ndarray): Q values for individual clusters
    """
    if not k:
       k = len(np.unique(k))

    # Select frames efficiently based on num_frames argument
    step = max(1, len(current_dists) // num_frames)  # Avoid division by zero
    selected_dists = current_dists[::step]

    cluster_qs = np.zeros(k, dtype=float)

    # Use parallel processing to speed up cluster computations
    cluster_qs = np.array(Parallel(n_jobs=n_jobs)(
        delayed(compute_rmse_for_cluster)(
            selected_dists[:, indices, :][:, :, indices],  # Extract distances for this cluster
            len(indices),
            use_max
        ) for i in range(k) if (indices := np.where(clustering == i)[0]).size > 0
    ))

    return np.mean(cluster_qs), cluster_qs


def get_Q_max_RMSE_wo_clustering(delta_matrices: np.ndarray):
  '''
  delta_matrices: List of distance matrices

  Returns the Q without the clusters
  '''
  rmses = np.sqrt(np.sum(np.sum(np.square(delta_matrices),axis=1),axis=1))
  return np.max(rmses)

def get_Q_max_RMSE_for_clustering(delta_matrices: np.ndarray, clustering: np.ndarray, k: int):
  '''
  delta_matrices: List of delta matrices
  clustering: Array containing index of cluster for each atom
  k: Amount of clusters

  Returns the Q and an array with the max RMSE of each cluster
  '''
  cluster_qs = np.zeros(k, dtype=np.ndarray)
  for i in range(k):
    indices = np.where(clustering == i)[0]
    if len(indices) == 1:
      cluster_qs[i] = 0
      continue
    cluster_deltas = delta_matrices[np.ix_(range(len(delta_matrices)), indices, indices)]
    rmses_for_cluster = np.sqrt(np.sum(np.sum(np.square(cluster_deltas),axis=1),axis=1))
    cluster_qs[i] = np.max(rmses_for_cluster)
  return np.sum(cluster_qs), cluster_qs

def get_RMSE(folded_dist: np.ndarray, current_dist: np.ndarray):
  return m.mean_squared_error(folded_dist, current_dist, squared=False)


# General cluster compare functions


import sklearn 
def get_clustering_similarity_true(true_clustering, clusterings, sim_function = sklearn.metrics.adjusted_rand_score):
    sim_results = []
    for i in range(len(clusterings)):
        sim_result = sim_function(true_clustering, clusterings[i])
        sim_results.append(sim_result)
    return sim_results

def get_clustering_similarity_true_trajectories(true_clusterings, clusterings, sim_function = sklearn.metrics.adjusted_rand_score):
    sim_results = []
    for i in range(len(clusterings)):
        sim_result = sim_function(true_clusterings[i], clusterings[i])
        sim_results.append(sim_result)
    return sim_result


def evaluate_clustering_nr_wrong_elements(true_labels, predicted_labels):

    cm = confusion_matrix(true_labels, predicted_labels)
    
    # Hungarian / Munkres algorithm
    row_ind, col_ind = linear_sum_assignment(-cm)
    
    correctly_clustered = cm[row_ind, col_ind].sum()
    
    total_items = len(true_labels)
    wrongly_clustered = total_items - correctly_clustered
    
    return wrongly_clustered




# modified / extended Q computation

import numpy as np
from itertools import combinations
from scipy.spatial import procrustes

def rmsd(P, Q):

    #return np.sqrt(np.mean(np.sum((P - Q) ** 2, axis=1)))
    #indeed equivalent
    diff = P - Q
    squared_diff = np.square(diff)
    
    
    sum_squared_diff = np.sum(squared_diff)
    
    
    rmse_for_frame_combis = np.sqrt(sum_squared_diff * (1 / len(P) ))
    return rmse_for_frame_combis

def get_Q_for_clustering_positions(positions: np.ndarray, clustering: np.ndarray, k=None, use_max=False, thinning_factor=None, center_positions = False, apply_superimposition=None, only_adjacent=False, noise_cluster=-1, ordered_clusters=True, divide_by_length_for_rmsd=True, for_loop=True, reference_frame = None, return_raw=False):
    '''
    positions: Array of positions with shape (N, T, 3) where N is the number of particles and T is the number of time frames
    clustering: Array containing index of cluster for each atom
    k: Amount of clusters

    Returns the Q and an array with the max RMSE of each cluster
    '''

    if not k:
        k_given = False
        k = len(np.unique(clustering))
        if noise_cluster:
            if noise_cluster in clustering:
                print("clustering contains noise")
                k = k - 1
    else:
        k_given = True

    cluster_qs = np.zeros(k)

    if thinning_factor:
        positions = positions[:, 0:positions.shape[1]:int(positions.shape[1] / thinning_factor), :]

    if ordered_clusters or k_given:
        k_range = range(k)
    else:
        k_range = np.unique(clustering)
        if noise_cluster:
            k_range = np.delete(k_range, np.where(k_range == noise_cluster))

    if return_raw:
        all_rmse_trajectories = []

    for i in k_range:
        indices = np.where(clustering == i)[0]


        if len(indices) <= 1:
            cluster_qs[i] = 0
            continue

        cluster_positions = positions[indices, :, :]


        if not for_loop:
            if not only_adjacent:
                combis = np.array(list(combinations(range(cluster_positions.shape[1]), 2)))
            else:
                combis = np.array([(t, t+1) for t in range(cluster_positions.shape[1] - 1)])
            if reference_frame is not None: 
                combis = [(reference_frame,t) for t in range(cluster_positions.shape[1] )]



            rmsd_for_frame_combis = np.zeros(len(combis))
            for j, (t1, t2) in enumerate(combis):


                P = cluster_positions[:, t1, :]
                Q = cluster_positions[:, t2, :]
                
                # Apply Procrustes analysis


                from scipy.spatial.transform import Rotation as R
                # Center the points
                if center_positions or apply_superimposition:
                    
                    P_mean = P.mean(axis=0)
                    Q_mean = Q.mean(axis=0)
                    P_centered = P - P_mean
                    Q_centered = Q - Q_mean
                    P = P_centered
                    Q = Q_centered

                if apply_superimposition == "kabsch": 
                    # Apply the Kabsch algorithm using scipy
                    rotation, _ = R.align_vectors(Q, P)
                    P_aligned = rotation.apply(P)
                    P = P_aligned
                elif apply_superimposition == "procrustes":
                    mtx1, mtx2, _ = procrustes(P, Q)
                    P = mtx1
                    Q = mtx2
                
                
                # Compute the RMSD
                rmsd_for_frame_combis[j] = rmsd(P, Q)
            
            if not use_max:
                cluster_qs[i] = np.mean(rmsd_for_frame_combis[rmsd_for_frame_combis != 0])
            else:
                cluster_qs[i] = np.max(rmsd_for_frame_combis[rmsd_for_frame_combis != 0])

        else:
            if not only_adjacent:
                combis = list(combinations(range(cluster_positions.shape[1]), 2))
            else:
                combis = [(t, t+1) for t in range(cluster_positions.shape[1] - 1)]
            if reference_frame is not None: 
                combis = [(reference_frame,t) for t in range(cluster_positions.shape[1])]


            rmsd_for_frame_combis = np.zeros(len(combis))

            for j, (t1, t2) in enumerate(combis):
                P = cluster_positions[:, t1, :]
                Q = cluster_positions[:, t2, :]
                
                # Apply Procrustes analysis

                from scipy.spatial.transform import Rotation as R
                # Center the points
                if center_positions or apply_superimposition:
                    
                    P_mean = P.mean(axis=0)
                    Q_mean = Q.mean(axis=0)
                    P_centered = P - P_mean
                    Q_centered = Q - Q_mean
                    P = P_centered
                    Q = Q_centered

                if apply_superimposition == "kabsch": 
                    # Apply the Kabsch algorithm using scipy
                    rotation, _ = R.align_vectors(Q, P)
                    P_aligned = rotation.apply(P)
                    P = P_aligned
                elif apply_superimposition == "procrustes":
                    mtx1, mtx2, _ = procrustes(P, Q)
                    P = mtx1
                    Q = mtx2
                
                # Compute the RMSD
                rmsd_for_frame_combis[j] = rmsd(P, Q)

            if not use_max:
                cluster_qs[i] = np.mean(rmsd_for_frame_combis[rmsd_for_frame_combis != 0])
            else:
                cluster_qs[i] = np.max(rmsd_for_frame_combis[rmsd_for_frame_combis != 0])
        if return_raw:
            all_rmse_trajectories.append(rmsd_for_frame_combis)

    if return_raw:
        return all_rmse_trajectories

    return np.mean(cluster_qs), cluster_qs




def get_Q_for_clustering_extended(current_dists: np.ndarray, clustering: np.ndarray, k = None, use_max = False, thinning_factor = None, only_adjacent = False, noise_cluster = -1, ordered_clusters = True, divide_by_length_for_rmsd=True, reference_frame=None, return_raw = False, for_loop = True):
  '''
  current_dists: List of distance matrices
  clustering: Array containing index of cluster for each atom
  k: Amount of clusters

  Returns the Q and an array with the max RMSE of each cluster
  '''

  if not k:
    k_given = False
    k = len(np.unique(clustering))
    if noise_cluster:
        if noise_cluster in clustering:
            print("clustering contains noise")
            k = k - 1
  else:
     k_given = True


  cluster_qs = np.zeros(k, dtype=np.ndarray)

  if thinning_factor:
    current_dists = current_dists[0:len(current_dists):int(len(current_dists)/thinning_factor)] 

  if ordered_clusters or k_given:
     k_range = range(k)
  else:
     k_range = np.unique(clustering)
     if noise_cluster:
        k_range = np.delete(k_range, np.where(k_range == noise_cluster))

  if return_raw:
    all_rmse_trajectories = []

  for i in k_range:
    indices = np.where(clustering == i)[0]

    if divide_by_length_for_rmsd:
        ca_count = len(indices)#*len(indices)
    else:
        ca_count = 1

    if len(indices) <= 1:
      cluster_qs[i] = 0
      continue
    cluster_dists = current_dists[np.ix_(range(len(current_dists)), indices, indices)]
    if not for_loop:
        
        if not only_adjacent:
            combis = np.array(list(combinations(cluster_dists, 2)))
        else:
           combis = []
           for ij in range(len(cluster_dists) - 1):
                
                delta1 = cluster_dists[ij]
                delta2 = cluster_dists[ij + 1]
                combis.append([delta1, delta2])

            # Convert list to numpy array for consistency
           combis = np.array(combis)
        if reference_frame is not None:
            combis = []
            for ij in range(len(cluster_dists) - 1):
                
                delta1 = cluster_dists[reference_frame]
                delta2 = cluster_dists[ij]
                combis.append([delta1, delta2])

            # Convert list to numpy array for consistency
            combis = np.array(combis)
           


        rmse_for_frame_combis = np.sqrt((np.sum(np.square(np.array([x[0] - x[1] for x in combis]).reshape(len(combis),-1)),axis=1))*(1/ca_count))
        
        if not use_max:
            cluster_qs[i] = np.mean(rmse_for_frame_combis[rmse_for_frame_combis != 0]) # ResiCon uses max
        else:
            cluster_qs[i] = np.max(rmse_for_frame_combis[rmse_for_frame_combis != 0]) # ResiCon uses max

    else:

        if not only_adjacent:
            combis = list(combinations(cluster_dists, 2))
        else:
            combis = []
            for ij in range(len(cluster_dists) - 1):
                
                delta1 = cluster_dists[ij]
                delta2 = cluster_dists[ij + 1]
                combis.append([delta1, delta2])
        if reference_frame is not None:
            combis = []
            for ij in range(len(cluster_dists) - 1):
                
                delta1 = cluster_dists[reference_frame]
                delta2 = cluster_dists[ij]
                combis.append([delta1, delta2])

            # Convert list to numpy array for consistency
            combis = np.array(combis)
           
        
        rmse_for_frame_combis = np.zeros(len(combis))

        for j, (delta1, delta2) in enumerate(combis):
            # Calculate difference and square it
            diff = delta1 - delta2
            squared_diff = np.square(diff)
            
            # Sum of squared differences
            sum_squared_diff = np.sum(squared_diff)
            
            # RMSE for this pair
            rmse_for_frame_combis[j] = np.sqrt(sum_squared_diff * (1 / ca_count ))


        if not use_max:
            cluster_qs[i] = np.mean(rmse_for_frame_combis[rmse_for_frame_combis != 0]) # ResiCon uses max
            
        else:
            cluster_qs[i] = np.max(rmse_for_frame_combis[rmse_for_frame_combis != 0]) # ResiCon uses max

    if return_raw:
        all_rmse_trajectories.append(rmse_for_frame_combis)

  if return_raw:
    return all_rmse_trajectories
        
  return np.mean(cluster_qs), cluster_qs




def get_Q_max_RMSE_for_clustering_extended(delta_matrices: np.ndarray, clustering: np.ndarray, k: int, for_loop=False, return_raw=False, thinning_factor = None):
  '''
  delta_matrices: List of delta matrices
  clustering: Array containing index of cluster for each atom
  k: Amount of clusters

  Returns the Q and an array with the max RMSE of each cluster
  '''


  if thinning_factor:
    delta_matrices = delta_matrices[0:len(delta_matrices):int(len(delta_matrices)/thinning_factor)] 

  
  if return_raw:
    all_rmse_trajectories = []

  cluster_qs = np.zeros(k, dtype=np.ndarray)
  for i in range(k):
    indices = np.where(clustering == i)[0]
    if len(indices) == 1:
      cluster_qs[i] = 0
      continue
    cluster_deltas = delta_matrices[np.ix_(range(len(delta_matrices)), indices, indices)]
    if not for_loop:
      rmses_for_cluster = np.sqrt(np.sum(np.sum(np.square(cluster_deltas),axis=1),axis=1))
      cluster_qs[i] = np.max(rmses_for_cluster)
    else:
      rmses_for_cluster = []
        
      for j in range(len(cluster_deltas)):
          sum_squared = 0
          for row in cluster_deltas[j]:
              for value in row:
                  sum_squared += value**2
          rmses_for_cluster.append(np.sqrt(sum_squared))
      
      # Convert list to numpy array for consistency
      rmses_for_cluster = np.array(rmses_for_cluster)
      
      # Store the maximum RMSE for the current cluster
      cluster_qs[i] = np.max(rmses_for_cluster)

    if return_raw:
       all_rmse_trajectories.append(rmses_for_cluster)

  if return_raw:
     return all_rmse_trajectories

  return np.mean(cluster_qs), cluster_qs



def rmsd(P, Q):
    """Computes RMSD between two sets of points."""
    return np.sqrt(np.mean(np.sum((P - Q) ** 2, axis=1)))


import numpy as np
from scipy.spatial import procrustes
from scipy.spatial.transform import Rotation as R

def rmsd(P, Q):
    """Computes RMSD between two sets of points."""
    return np.sqrt(np.mean(np.sum((P - Q) ** 2, axis=1)))

def get_RMSD_to_average_structure(positions: np.ndarray, clustering: np.ndarray, k=None, apply_superimposition=None, center_positions=False, return_raw=False):
    '''
    positions: Array of positions with shape (T, nr_objects, 3) where T is the number of time frames (steps), 
               nr_objects is the number of particles, and 3 is for x, y, z coordinates.
    clustering: Array containing index of cluster for each object (of length nr_objects)
    k: Number of clusters

    Returns the RMSD for each cluster at each time step relative to the superimposed average structure.
    '''

    if not k:
        k = len(np.unique(clustering))

    # Initialize array to store RMSD for each cluster at each time step
    rmsd_per_timestep = np.zeros((k, positions.shape[0]))  # Shape (k, T)

    # Iterate over each cluster
    for i in range(k):
        # Get the indices of the objects in cluster i
        indices = np.where(clustering == i)[0]

        if len(indices) <= 1:
            rmsd_per_timestep[i, :] = 0
            continue

        # Extract positions for this cluster (Shape: T, len(indices), 3)
        cluster_positions = positions[:, indices, :]

        # Compute the average structure (mean position across all time steps) (Shape: len(indices), 3)
        average_structure = cluster_positions.mean(axis=0)  # Shape (len(indices), 3)

        # Optionally center the positions around the average structure
        if center_positions:
            average_structure_mean = average_structure.mean(axis=0)
            average_structure -= average_structure_mean
            cluster_positions -= cluster_positions.mean(axis=1, keepdims=True)

        # Optionally apply superimposition to the average structure
        if apply_superimposition == "kabsch":
            # Align the cluster positions to the average structure using Kabsch
            for t in range(positions.shape[0]):  # Iterate over time steps
                rotation, _ = R.align_vectors(average_structure, cluster_positions[t, :, :])
                aligned_positions = rotation.apply(cluster_positions[t, :, :])
                rmsd_per_timestep[i, t] = rmsd(aligned_positions, average_structure)

        elif apply_superimposition == "procrustes":
            for t in range(positions.shape[0]):
                mtx1, mtx2, _ = procrustes(average_structure, cluster_positions[t, :, :])
                rmsd_per_timestep[i, t] = rmsd(mtx1, mtx2)
        else:
            # If no superimposition, calculate RMSD directly
            for t in range(positions.shape[0]):
                rmsd_per_timestep[i, t] = rmsd(cluster_positions[t, :, :], average_structure)

    if return_raw:
        return rmsd_per_timestep

    # Optionally return the mean RMSD for each cluster across all time steps
    return np.mean(rmsd_per_timestep, axis=1), rmsd_per_timestep



def rmsd(P, Q):
    """Computes RMSD between two sets of points."""
    return np.sqrt(np.mean(np.sum((P - Q) ** 2, axis=1)))


def relabel_to_zero(clustering):
    # Original array
    arr = np.array(clustering).copy()

    # Get the unique values, sort them, and create a mapping
    unique_values = sorted(np.unique(arr))

    # Create a mapping of original values to the new labels
    mapping = {value: idx for idx, value in enumerate(unique_values)}

    # Apply the mapping to the array
    relabeled_array = np.vectorize(mapping.get)(arr)
    return relabeled_array, mapping

def get_RMSD_to_reference(positions: np.ndarray, clustering: np.ndarray, k=None, reference_frame=None, 
                           apply_superimposition=None, center_positions=True, return_raw=False, 
                           start=None, end=None, relabel=False):
    '''
    positions: Array of positions with shape (T, nr_objects, 3) where T is the number of time frames (steps), 
               nr_objects is the number of particles, and 3 is for x, y, z coordinates.
    clustering: Array containing index of cluster for each object (of length nr_objects)
    k: Number of clusters
    reference_frame: Specific time frame (index) to use as the reference structure. If None, the average structure will be used.
    apply_superimposition: Superimposition method to use ('kabsch', 'procrustes', or None).
    center_positions: Whether to center the positions around the reference structure.
    return_raw: Whether to return the RMSD values for each time step or only the average RMSD.
    start: The starting time frame (inclusive). If None, all frames are considered.
    end: The ending time frame (inclusive). If None, all frames are considered.
    
    Returns the RMSD for each cluster at each time step relative to the reference structure (either a specific frame or the average).
    '''
    
    if not k:
        k = len(np.unique(clustering))

    if relabel:
        clustering, mapping = relabel_to_zero(clustering)
        print(mapping)

    # Determine the valid time range to consider
    if start is not None and end is not None:
        time_range = range(start, end + 1)
    else:
        time_range = range(positions.shape[0])  # Consider all time frames

    # Initialize array to store RMSD for each cluster at each selected time step
    rmsd_per_timestep = np.zeros((k, len(time_range)))  # Shape (k, number of frames in the range)

    # Iterate over each cluster
    for i in range(k):
        # Get the indices of the objects in cluster i
        indices = np.where(clustering == i)[0]

        if len(indices) <= 1:
            rmsd_per_timestep[i, :] = 0
            continue

        # Extract positions for this cluster (Shape: T, len(indices), 3)
        cluster_positions = positions[:, indices, :]

        # Determine the reference structure
        if reference_frame is not None:
            # Use the specific frame as the reference structure (Shape: len(indices), 3)
            reference_structure = cluster_positions[reference_frame, :, :]
        else:
            # Compute the average structure (mean position across all time steps) (Shape: len(indices), 3)
            reference_structure = cluster_positions.mean(axis=0)  # Shape (len(indices), 3)

        # Optionally center the positions around the reference structure
        if center_positions:
            # Center the cluster positions
            cluster_positions -= cluster_positions.mean(axis=1, keepdims=True)

            # Center the reference structure as well
            reference_structure -= reference_structure.mean(axis=0)

        # Optionally apply superimposition to the reference structure
        if apply_superimposition == "kabsch":
            for t_idx, t in enumerate(time_range):  # Iterate only over the selected time steps
                rotation, _ = R.align_vectors(reference_structure, cluster_positions[t, :, :])
                aligned_positions = rotation.apply(cluster_positions[t, :, :])
                rmsd_per_timestep[i, t_idx] = rmsd(aligned_positions, reference_structure)

        elif apply_superimposition == "procrustes":
            for t_idx, t in enumerate(time_range):
                mtx1, mtx2, _ = procrustes(reference_structure, cluster_positions[t, :, :])
                rmsd_per_timestep[i, t_idx] = rmsd(mtx1, mtx2)
        else:
            # If no superimposition, calculate RMSD directly
            for t_idx, t in enumerate(time_range):
                rmsd_per_timestep[i, t_idx] = rmsd(cluster_positions[t, :, :], reference_structure)

    if return_raw:
        return rmsd_per_timestep

    # Optionally return the mean RMSD for each cluster across the selected time steps
    return np.mean(rmsd_per_timestep, axis=1), rmsd_per_timestep

def max_overlap_matching(timestep_clusters):
    """
    Maximize the overlap between clustering labels across consecutive time steps using the Hungarian algorithm.

    Args:
    timestep_clusters (list): List of dictionaries containing 'start', 'end', and 'clustering' arrays.

    Returns:
    List: A list of dictionaries with updated clustering labels after applying the Hungarian algorithm.
    """
    updated_clusters = []

    # Iterate through each pair of consecutive time steps
    for i in range(len(timestep_clusters) - 1):
        # Get the current and next time step clusters
        curr_start, curr_end, curr_clustering = timestep_clusters[i]['start'], timestep_clusters[i]['end'], timestep_clusters[i]['clustering']
        next_start, next_end, next_clustering = timestep_clusters[i + 1]['start'], timestep_clusters[i + 1]['end'], timestep_clusters[i + 1]['clustering']

        # Get the number of clusters in the current and next time step
        num_curr_clusters = len(np.unique(curr_clustering))
        num_next_clusters = len(np.unique(next_clustering))

        # Create a cost matrix where the rows are current clusters and columns are next clusters
        cost_matrix = np.zeros((num_curr_clusters, num_next_clusters), dtype=int)

        # Populate the cost matrix with the number of overlapping objects between each pair of clusters
        for curr_label in range(num_curr_clusters):
            for next_label in range(num_next_clusters):
                # Calculate the overlap (number of objects with the same label in both clusters)
                overlap = np.sum((curr_clustering == curr_label) & (next_clustering == next_label))
                cost_matrix[curr_label, next_label] = -overlap  # Negative because we want to maximize overlap

        # Use the Hungarian algorithm to find the optimal assignment
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # Create a mapping from the current clusters to the next clusters
        label_mapping = dict(zip(row_ind, col_ind))

        # Update the clustering labels for the next time step based on the mapping
        next_clustering_updated = np.copy(next_clustering)
        for curr_label, next_label in label_mapping.items():
            next_clustering_updated[next_clustering == next_label] = curr_label

        # Store the updated clustering in the result list
        updated_clusters.append({
            'start': curr_start,
            'end': curr_end,
            'clustering': curr_clustering
        })

        # Update the next step with the new clustering labels
        timestep_clusters[i + 1]['clustering'] = next_clustering_updated

    # Add the last cluster (as it doesn't have a next step to compare with)
    updated_clusters.append(timestep_clusters[-1])

    return updated_clusters


def relabel_clustering_starting_from_zero(clustering):
    # Original array
    arr = np.array(clustering).copy()

    # Get the unique values, sort them, and create a mapping
    unique_values = sorted(np.unique(arr))

    # Create a mapping of original values to the new labels
    mapping = {value: idx for idx, value in enumerate(unique_values)}

    # Apply the mapping to the array
    relabeled_array = np.vectorize(mapping.get)(arr)
    return relabeled_array


def get_sdm_batches(summed_delta_matrices,get_total_rmses_for_frames,batch_size = 10):
    batched_summed_delta_matrices = []
    batched_get_total_rmses_for_frames = []
    for i in range(0, len(summed_delta_matrices), batch_size):
        try:
            batched_summed_delta_matrices.append(np.mean(summed_delta_matrices[i:i+batch_size]))
            batched_get_total_rmses_for_frames.append(np.mean(get_total_rmses_for_frames[i:i+batch_size]))
        except:
            pass
    return batched_summed_delta_matrices, batched_get_total_rmses_for_frames