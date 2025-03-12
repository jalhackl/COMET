import math
import CONSTANTS
import mdtraj as md
import distance_matrix as dm
import manipulate_trajectory as mt
import numpy as np
import multiprocessing
import warnings
import compare_clusterings as cc

from functools import wraps
from time import time


def preprocess_protein_trajectory(prot_info, k_cluster=None):
  frames_count = prot_info[3]
  traj_array, k_cluster = preprocessing(prot_info,frames_count,k_cluster)
  return traj_array, k_cluster


def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        #print('func:%r took: %2.4f sec' % (f.__name__, te-ts))
        return result
    return wrap

warnings.filterwarnings("ignore")
pool = multiprocessing.Pool()


@timing
def preprocessing(prot_info,frames_count,k_cluster, calculate_clusters = True):

    if prot_info[1] != None:
      traj = md.load(CONSTANTS.path_prefix + prot_info[2] + prot_info[0], top=CONSTANTS.path_prefix + prot_info[2] + prot_info[1])
    else:
      traj = md.load(CONSTANTS.path_prefix + prot_info[2] + prot_info[0])

    traj = mt.restrict_protein_to_elements(traj, 'CA').center_coordinates()     # Only consider CA-atoms
    traj_array = traj.xyz[0:frames_count]   # Trajectory to array
    if k_cluster is None:
      k_cluster = int(math.sqrt(traj.n_atoms))

    if calculate_clusters:
       k_cluster = int(math.sqrt(traj.n_atoms))

    return traj_array, k_cluster


def preprocessing_return_traj(prot_info,frames_count,k_cluster, calculate_clusters = True):
    print(prot_info[0])

    if prot_info[1] != None:
      traj = md.load(CONSTANTS.path_prefix + prot_info[2] + prot_info[0], top=CONSTANTS.path_prefix + prot_info[2] + prot_info[1])
    elif prot_info[1] == None and prot_info[0].split(".")[-1]=="dcd":
            traj = md.load_dcd(CONSTANTS.path_prefix + prot_info[2] + prot_info[0])
    else:
      traj = md.load(CONSTANTS.path_prefix + prot_info[2] + prot_info[0])

    traj = mt.restrict_protein_to_elements(traj, 'CA').center_coordinates()     # Only consider CA-atoms
    traj_array = traj.xyz[0:frames_count]   # Trajectory to array
    if k_cluster is None:
      k_cluster = int(math.sqrt(traj.n_atoms))

    if calculate_clusters:
       k_cluster = int(math.sqrt(traj.n_atoms))

    return traj, traj_array, k_cluster


@timing
def get_distance_matrices(traj_array):
    return np.array(list(pool.map(dm.calculate_distance_matrix, traj_array)))

@timing
def get_delta_matrices_no_chunk(dist_matrices):
    return np.absolute(np.diff(dist_matrices,axis=0))

@timing
def get_delta_matrices(dist_matrices):
    n, h, w = dist_matrices.shape
    chunk_size = 1000  # Define a suitable chunk size
    for i in range(0, n - 1, chunk_size):
        end = min(i + chunk_size, n - 1)
        # Compute the differences for the current chunk
        diff_chunk = np.diff(dist_matrices[i:end + 1], axis=0)
        # Take the absolute value and store it back in the original array
        dist_matrices[i:end] = np.absolute(diff_chunk)
    return dist_matrices[:n - 1]

@timing
def calculate_average_delta_matrix(delta_matrices):
    return np.sum(delta_matrices, axis=0)/len(delta_matrices)


#new
@timing
def get_var_matrices_no_chunk(dist_matrices):
    return np.absolute(np.var(dist_matrices,axis=0))

@timing
def get_var_matrices(dist_matrices):
    n, h, w = dist_matrices.shape
    chunk_size = 1000  # Define a suitable chunk size

    # Step 1: Calculate the mean in chunks
    mean_matrix = np.zeros((h, w), dtype=dist_matrices.dtype)
    for i in range(0, n, chunk_size):
        end = min(i + chunk_size, n)
        mean_matrix += np.sum(dist_matrices[i:end], axis=0)
    mean_matrix /= n

    # Step 2: Calculate the squared deviations from the mean in chunks
    for i in range(0, n, chunk_size):
        end = min(i + chunk_size, n)
        dist_matrices[i:end] -= mean_matrix
        dist_matrices[i:end] **= 2

    # Step 3: Calculate the variance in chunks
    var_matrix = np.zeros((h, w), dtype=dist_matrices.dtype)
    for i in range(0, n, chunk_size):
        end = min(i + chunk_size, n)
        var_matrix += np.sum(dist_matrices[i:end], axis=0)
    var_matrix /= n

    # Step 4: Take the absolute value (if necessary) and store it back in dist_matrices
    var_matrix = np.absolute(var_matrix)
    dist_matrices[:n - 1] = var_matrix

    return var_matrix


@timing
def get_std_matrices_no_chunk(dist_matrices):
    return np.absolute(np.std(dist_matrices,axis=0))


@timing
def get_std_matrices(dist_matrices):
    n, h, w = dist_matrices.shape
    chunk_size = 1000  # Define a suitable chunk size

    # Step 1: Calculate the mean in chunks
    mean_matrix = np.zeros((h, w), dtype=dist_matrices.dtype)
    for i in range(0, n, chunk_size):
        end = min(i + chunk_size, n)
        mean_matrix += np.sum(dist_matrices[i:end], axis=0)
    mean_matrix /= n

    # Step 2: Calculate the squared deviations from the mean in chunks
    for i in range(0, n, chunk_size):
        end = min(i + chunk_size, n)
        dist_matrices[i:end] -= mean_matrix
        dist_matrices[i:end] **= 2

    # Step 3: Calculate the variance in chunks
    var_matrix = np.zeros((h, w), dtype=dist_matrices.dtype)
    for i in range(0, n, chunk_size):
        end = min(i + chunk_size, n)
        var_matrix += np.sum(dist_matrices[i:end], axis=0)
    var_matrix /= n

    # Step 4: Calculate the standard deviation
    std_matrix = np.sqrt(var_matrix)
    
    # Step 5: Take the absolute value (if necessary) and store it back in dist_matrices
    std_matrix = np.absolute(std_matrix)
    dist_matrices[:n - 1] = std_matrix

    return std_matrix

#new, not used
@timing
def calculate_var_delta_matrix(delta_matrices):
    return np.var(delta_matrices, axis=0)/len(delta_matrices)

@timing
def main(prot_info,k_cluster,assign_labels='discretize', plot_delta_heatmaps=True, chunk_delta=True) -> np.ndarray:

  frames_count = prot_info[3]
  traj_array, k_cluster = preprocessing(prot_info,frames_count,k_cluster)
  
  dist_matrices = get_distance_matrices(traj_array)
  del traj_array
  if chunk_delta:
    delta_matrices = get_delta_matrices(dist_matrices)
  else:
     delta_matrices = get_delta_matrices_no_chunk(dist_matrices)
  #del dist_matrices
  average_delta_matrix = calculate_average_delta_matrix(delta_matrices)

  print("avg.shape")
  #print(average_delta_matrix.shape)
  print(average_delta_matrix)
  clustering = dm.spectral_clustering_on_deltas(average_delta_matrix, k_cluster, assign_labels, plot_delta_heatmaps=plot_delta_heatmaps)
  Q, _ = cc.get_Q_for_clustering(dist_matrices, clustering, k_cluster)

  clustering2 = dm.spectral_clustering_on_deltas(average_delta_matrix, k_cluster, 'kmeans', plot_delta_heatmaps=plot_delta_heatmaps)
  clustering3 = dm.agglomerative_clustering_on_deltas(average_delta_matrix, k_cluster, 'average')
  clustering4 = dm.agglomerative_clustering_on_deltas(average_delta_matrix, k_cluster, 'complete')
  clustering5 = dm.agglomerative_clustering_on_deltas(average_delta_matrix, k_cluster, 'single')
  #clustering6 = dm.agglomerative_clustering_on_deltas(average_delta_matrix, k_cluster, 'ward')
  Q2, _ = cc.get_Q_for_clustering(dist_matrices, clustering2, k_cluster)
  Q3, _ = cc.get_Q_for_clustering(dist_matrices, clustering3, k_cluster)
  Q4, _ = cc.get_Q_for_clustering(dist_matrices, clustering4, k_cluster)
  Q5, _ = cc.get_Q_for_clustering(dist_matrices, clustering5, k_cluster)
  #Q6, _ = cc.get_Q_for_clustering(dist_matrices, clustering6, k_cluster)
  return clustering, Q, Q2, Q3, Q4, Q5

#what is this?
@timing
def get_delta_matrices_clustering(prot_info,k_cluster,assign_labels='discretize', clustering_algorithm="spectral", no_q_computation=False, eigengap = False, silhouette=False, min_cluster_size=2, min_samples=1, plot_delta_heatmaps=True, chunk_delta=True) -> np.ndarray:

  frames_count = prot_info[3]
  traj_array, k_cluster = preprocessing(prot_info,frames_count,k_cluster)
  
  dist_matrices = get_distance_matrices(traj_array)
  del traj_array
  if chunk_delta:
    delta_matrices = get_delta_matrices(dist_matrices)
  else:
    delta_matrices = get_delta_matrices_no_chunk(dist_matrices)

  return dist_matrices, delta_matrices


@timing
def main_clustering(prot_info,k_cluster,assign_labels='discretize', clustering_algorithm="spectral", no_q_computation=False, eigengap = False, silhouette=False, min_cluster_size=2, min_samples=1, plot_delta_heatmaps=True, chunk_delta=True) -> np.ndarray:

  frames_count = prot_info[3]
  traj_array, k_cluster = preprocessing(prot_info,frames_count,k_cluster)
  
  dist_matrices = get_distance_matrices(traj_array)
  del traj_array
  if chunk_delta:
    delta_matrices = get_delta_matrices(dist_matrices)
  else:
     delta_matrices = get_delta_matrices_no_chunk(dist_matrices)
  #del dist_matrices
  average_delta_matrix = calculate_average_delta_matrix(delta_matrices)

  print("avg.shape")
  #print(average_delta_matrix.shape)
  #print("before average delta matrix")
  print(average_delta_matrix)
  #print("after average delta matrix")
  matrix = []

  if clustering_algorithm == "spectral":
    clustering = dm.spectral_clustering_on_deltas(average_delta_matrix, k_cluster, assign_labels, eigengap = eigengap, silhouette = silhouette, plot_delta_heatmaps=plot_delta_heatmaps)
  elif clustering_algorithm == "hdbscan":
     clustering, k_cluster, matrix = dm.hdbscan_clustering_on_deltas(average_delta_matrix, k_cluster, assign_labels, min_cluster_size=min_cluster_size, min_samples=min_samples)
  elif clustering_algorithm == "agglomerative":
     clustering, k_cluster, matrix = dm.agglomerative_clustering_on_deltas_ward(average_delta_matrix, None)


  Q = []
  

  if not no_q_computation: 
    try:
      Q, _ = cc.get_Q_for_clustering(dist_matrices, clustering, k_cluster)
    except:
      print("something wrong with clustering Q computation")

  
  '''
  clustering2 = dm.spectral_clustering_on_deltas(average_delta_matrix, k_cluster, 'kmeans')
  clustering3 = dm.agglomerative_clustering_on_deltas(average_delta_matrix, k_cluster, 'average')
  clustering4 = dm.agglomerative_clustering_on_deltas(average_delta_matrix, k_cluster, 'complete')
  clustering5 = dm.agglomerative_clustering_on_deltas(average_delta_matrix, k_cluster, 'single')
  #clustering6 = dm.agglomerative_clustering_on_deltas(average_delta_matrix, k_cluster, 'ward')
  Q2, _ = cc.get_Q_for_clustering(dist_matrices, clustering2, k_cluster)
  Q3, _ = cc.get_Q_for_clustering(dist_matrices, clustering3, k_cluster)
  Q4, _ = cc.get_Q_for_clustering(dist_matrices, clustering4, k_cluster)
  Q5, _ = cc.get_Q_for_clustering(dist_matrices, clustering5, k_cluster)
  #Q6, _ = cc.get_Q_for_clustering(dist_matrices, clustering6, k_cluster)
  '''
  return clustering, Q, matrix



def main_var(prot_info,k_cluster,assign_labels='discretize', plot_delta_heatmaps=True, chunk_delta=True) -> np.ndarray:

  frames_count = prot_info[3]
  traj_array, k_cluster = preprocessing(prot_info,frames_count,k_cluster)
  
  dist_matrices = get_distance_matrices(traj_array)
  del traj_array
  if chunk_delta:
    delta_matrices = get_delta_matrices(dist_matrices)
  else:
     delta_matrices = get_delta_matrices_no_chunk(dist_matrices)
  #del dist_matrices
  average_delta_matrix = calculate_average_delta_matrix(delta_matrices)

  #new
  #var_delta_matrix = calculate_var_delta_matrix(delta_matrices)
  if chunk_delta:
      var_delta_matrix = get_var_matrices(dist_matrices)
  else:
      var_delta_matrix = get_var_matrices_no_chunk(dist_matrices)
  #average_delta_matrix = np.stack([average_delta_matrix, var_delta_matrix])

  clustering = dm.spectral_clustering_on_deltas_var(average_delta_matrix,var_delta_matrix, k_cluster, assign_labels)
  Q, _ = cc.get_Q_for_clustering(dist_matrices, clustering, k_cluster)
  

  clustering2 = dm.spectral_clustering_on_deltas(average_delta_matrix, k_cluster, 'kmeans')
  clustering3 = dm.agglomerative_clustering_on_deltas(average_delta_matrix, k_cluster, 'average')
  clustering4 = dm.agglomerative_clustering_on_deltas(average_delta_matrix, k_cluster, 'complete')
  clustering5 = dm.agglomerative_clustering_on_deltas(average_delta_matrix, k_cluster, 'single')
  #clustering6 = dm.agglomerative_clustering_on_deltas(average_delta_matrix, k_cluster, 'ward')
  Q2, _ = cc.get_Q_for_clustering(dist_matrices, clustering2, k_cluster)
  Q3, _ = cc.get_Q_for_clustering(dist_matrices, clustering3, k_cluster)
  Q4, _ = cc.get_Q_for_clustering(dist_matrices, clustering4, k_cluster)
  Q5, _ = cc.get_Q_for_clustering(dist_matrices, clustering5, k_cluster)
  #Q6, _ = cc.get_Q_for_clustering(dist_matrices, clustering6, k_cluster)
  return clustering, Q, Q2, Q3, Q4, Q5


def main_clustring_var(prot_info,k_cluster,assign_labels='discretize', no_q_computation=False, plot_delta_heatmaps=True, chunk_delta=True) -> np.ndarray:

  frames_count = prot_info[3]
  traj_array, k_cluster = preprocessing(prot_info,frames_count,k_cluster)
  
  dist_matrices = get_distance_matrices(traj_array)
  del traj_array
  if chunk_delta:
    delta_matrices = get_delta_matrices(dist_matrices)
  else:
     delta_matrices = get_delta_matrices_no_chunk(dist_matrices)
  
  average_delta_matrix = calculate_average_delta_matrix(delta_matrices)

  #new
  #var_delta_matrix = calculate_var_delta_matrix(delta_matrices)
  #var_delta_matrix = get_var_matrices(dist_matrices)
  if chunk_delta:
      var_delta_matrix = get_var_matrices(dist_matrices)
  else:
      var_delta_matrix = get_var_matrices_no_chunk(dist_matrices)
  #average_delta_matrix = np.stack([average_delta_matrix, var_delta_matrix])

  clustering = dm.spectral_clustering_on_deltas_var(average_delta_matrix,var_delta_matrix, k_cluster, assign_labels)
  Q, _ = cc.get_Q_for_clustering(dist_matrices, clustering, k_cluster)

  
  Q = []
  

  if not no_q_computation: 
    try:
      Q, _ = cc.get_Q_for_clustering(dist_matrices, clustering, k_cluster)
    except:
      print("something wrong with clustering Q computation")

  '''
  clustering2 = dm.spectral_clustering_on_deltas(average_delta_matrix, k_cluster, 'kmeans')
  clustering3 = dm.agglomerative_clustering_on_deltas(average_delta_matrix, k_cluster, 'average')
  clustering4 = dm.agglomerative_clustering_on_deltas(average_delta_matrix, k_cluster, 'complete')
  clustering5 = dm.agglomerative_clustering_on_deltas(average_delta_matrix, k_cluster, 'single')
  #clustering6 = dm.agglomerative_clustering_on_deltas(average_delta_matrix, k_cluster, 'ward')
  Q2, _ = cc.get_Q_for_clustering(dist_matrices, clustering2, k_cluster)
  Q3, _ = cc.get_Q_for_clustering(dist_matrices, clustering3, k_cluster)
  Q4, _ = cc.get_Q_for_clustering(dist_matrices, clustering4, k_cluster)
  Q5, _ = cc.get_Q_for_clustering(dist_matrices, clustering5, k_cluster)
  #Q6, _ = cc.get_Q_for_clustering(dist_matrices, clustering6, k_cluster)
  '''
  return clustering, Q


def main_clustering_varadd(prot_info,k_cluster,assign_labels='discretize', no_q_computation=False, fuser_alg = "agreement", silhouette=False, plot_delta_heatmaps=True, chunk_delta=True) -> np.ndarray:

  frames_count = prot_info[3]
  traj_array, k_cluster = preprocessing(prot_info,frames_count,k_cluster)
  
  dist_matrices = get_distance_matrices(traj_array)
  del traj_array
  if chunk_delta:
      delta_matrices = get_delta_matrices(dist_matrices)
  else:
     delta_matrices = get_delta_matrices_no_chunk(dist_matrices)

  average_delta_matrix = calculate_average_delta_matrix(delta_matrices)

  #new
  #var_delta_matrix = calculate_var_delta_matrix(delta_matrices)
  if chunk_delta:
      std_delta_matrix = get_std_matrices(dist_matrices)
  else:
      std_delta_matrix = get_std_matrices_no_chunk(dist_matrices)
  #average_delta_matrix = np.stack([average_delta_matrix, var_delta_matrix])

  clustering = dm.spectral_clustering_on_deltas_varadd(average_delta_matrix,std_delta_matrix, k_cluster, assign_labels, fuser_alg=fuser_alg, silhouette=silhouette, plot_delta_heatmaps=plot_delta_heatmaps)
  Q, _ = cc.get_Q_for_clustering(dist_matrices, clustering, k_cluster)

  
  Q = []
  

  if not no_q_computation: 
    try:
      Q, _ = cc.get_Q_for_clustering(dist_matrices, clustering, k_cluster)
    except:
      print("something wrong with clustering Q computation")

  '''
  clustering2 = dm.spectral_clustering_on_deltas(average_delta_matrix, k_cluster, 'kmeans')
  clustering3 = dm.agglomerative_clustering_on_deltas(average_delta_matrix, k_cluster, 'average')
  clustering4 = dm.agglomerative_clustering_on_deltas(average_delta_matrix, k_cluster, 'complete')
  clustering5 = dm.agglomerative_clustering_on_deltas(average_delta_matrix, k_cluster, 'single')
  #clustering6 = dm.agglomerative_clustering_on_deltas(average_delta_matrix, k_cluster, 'ward')
  Q2, _ = cc.get_Q_for_clustering(dist_matrices, clustering2, k_cluster)
  Q3, _ = cc.get_Q_for_clustering(dist_matrices, clustering3, k_cluster)
  Q4, _ = cc.get_Q_for_clustering(dist_matrices, clustering4, k_cluster)
  Q5, _ = cc.get_Q_for_clustering(dist_matrices, clustering5, k_cluster)
  #Q6, _ = cc.get_Q_for_clustering(dist_matrices, clustering6, k_cluster)
  '''
  return clustering, Q


def run_experiments(protein_info_list):
  aQ = []
  aQ2 = []
  aQ3 = []
  aQ4 = []
  aQ5 = []
  #aQ6 = []
  for prot_info in protein_info_list:
    print(prot_info[0])
    _,Q,Q2,Q3,Q4,Q5 = main(prot_info, k_cluster=prot_info[4], assign_labels='discretize')
    aQ.append(Q)
    aQ2.append(Q2)
    aQ3.append(Q3)
    aQ4.append(Q4)
    aQ5.append(Q5)
    #aQ6.append(Q6)
  
  print('Disc: ')
  print(str(np.mean(np.array(aQ))))
  print(str(np.std(np.array(aQ))))
  print('K Means: ')
  print(str(np.mean(np.array(aQ2))))
  print(str(np.std(np.array(aQ2))))
  print('Average Link:')
  print(str(np.mean(np.array(aQ3))))
  print(str(np.std(np.array(aQ3))))
  print('Complete Link:')
  print(str(np.mean(np.array(aQ4))))
  print(str(np.std(np.array(aQ4))))
  print('Single Link:')
  print(str(np.mean(np.array(aQ5))))
  print(str(np.std(np.array(aQ5))))
  #print('Mean Ward: ' + str(np.mean(np.array(aQ6))))
  #print('STD Ward: ' + str(np.std(np.array(aQ6))))

#un_experiments(CONSTANTS.tr)
'''
run_experiments(CONSTANTS.mcg)

run_experiments(CONSTANTS.sav)
run_experiments(CONSTANTS.r1)
run_experiments(CONSTANTS.abeta)
'''

#if __name__ == '__main__':
#    main()


def run_experiments_return(protein_info_list, assign_labels='discretize'):
  aQ = []
  aQ2 = []
  aQ3 = []
  aQ4 = []
  aQ5 = []
  #aQ6 = []

  clusterings = []
  for prot_info in protein_info_list:
    print(prot_info[0])
    _,Q,Q2,Q3,Q4,Q5 = main(prot_info, k_cluster=prot_info[4], assign_labels=assign_labels)
    aQ.append(Q)
    aQ2.append(Q2)
    aQ3.append(Q3)
    aQ4.append(Q4)
    aQ5.append(Q5)
    #aQ6.append(Q6)
  
    clusterings.append(_)
  print('Disc: ')
  print(str(np.mean(np.array(aQ))))
  print(str(np.std(np.array(aQ))))
  print('K Means: ')
  print(str(np.mean(np.array(aQ2))))
  print(str(np.std(np.array(aQ2))))
  print('Average Link:')
  print(str(np.mean(np.array(aQ3))))
  print(str(np.std(np.array(aQ3))))
  print('Complete Link:')
  print(str(np.mean(np.array(aQ4))))
  print(str(np.std(np.array(aQ4))))
  print('Single Link:')
  print(str(np.mean(np.array(aQ5))))
  print(str(np.std(np.array(aQ5))))
  #print('Mean Ward: ' + str(np.mean(np.array(aQ6))))
  #print('STD Ward: ' + str(np.std(np.array(aQ6))))

  return clusterings, aQ, aQ2, aQ3, aQ4, aQ5




def run_experiments_return_var(protein_info_list, assign_labels='discretize'):
  aQ = []
  aQ2 = []
  aQ3 = []
  aQ4 = []
  aQ5 = []
  #aQ6 = []

  clusterings = []
  for prot_info in protein_info_list:
    print(prot_info[0])
    _,Q,Q2,Q3,Q4,Q5 = main_var(prot_info, k_cluster=prot_info[4], assign_labels=assign_labels)
    aQ.append(Q)
    aQ2.append(Q2)
    aQ3.append(Q3)
    aQ4.append(Q4)
    aQ5.append(Q5)
    #aQ6.append(Q6)
  
    clusterings.append(_)
  print('Disc: ')
  print(str(np.mean(np.array(aQ))))
  print(str(np.std(np.array(aQ))))
  print('K Means: ')
  print(str(np.mean(np.array(aQ2))))
  print(str(np.std(np.array(aQ2))))
  print('Average Link:')
  print(str(np.mean(np.array(aQ3))))
  print(str(np.std(np.array(aQ3))))
  print('Complete Link:')
  print(str(np.mean(np.array(aQ4))))
  print(str(np.std(np.array(aQ4))))
  print('Single Link:')
  print(str(np.mean(np.array(aQ5))))
  print(str(np.std(np.array(aQ5))))
  #print('Mean Ward: ' + str(np.mean(np.array(aQ6))))
  #print('STD Ward: ' + str(np.std(np.array(aQ6))))

  return clusterings, aQ, aQ2, aQ3, aQ4, aQ5

def load_pdb(prot_info,frames_count,k_cluster):
    traj = md.load(CONSTANTS.path_prefix + prot_info[2] + prot_info[0], top=CONSTANTS.path_prefix + prot_info[2] + prot_info[1])


    return traj

def run_it(assign_labels='discretize'):
  print("firstexperimentr")
  clusterings, aQ, aQ2, aQ3, aQ4, aQ5 = run_experiments_return(CONSTANTS.mcg, assign_labels)

  return clusterings, aQ, aQ2, aQ3, aQ4, aQ5

def run_it_sims(sims, assign_labels='discretize'):
  print("firstexperimentsims")
  clusterings, aQ, aQ2, aQ3, aQ4, aQ5 = run_experiments_return(sims, assign_labels)

  return clusterings, aQ, aQ2, aQ3, aQ4, aQ5

def run_it_sims_var(sims, assign_labels='discretize'):
  print("firstexperimentsimsvar")
  clusterings, aQ, aQ2, aQ3, aQ4, aQ5 = run_experiments_return_var(sims, assign_labels)

  return clusterings, aQ, aQ2, aQ3, aQ4, aQ5



def run_it_chig():
  print("firstexperimentr")
  clusterings, aQ, aQ2, aQ3, aQ4, aQ5 = run_experiments_return(CONSTANTS.chig)

  return clusterings, aQ, aQ2, aQ3, aQ4, aQ5


def run_it_var(assign_labels='discretize'):
  print("firstexperimentr_var")
  clusterings, aQ, aQ2, aQ3, aQ4, aQ5 = run_experiments_return_var(CONSTANTS.mcg, assign_labels)

  return clusterings, aQ, aQ2, aQ3, aQ4, aQ5

def run_load():
   traj = load_pdb(CONSTANTS.mcg)

   return traj


def run_clustering(protein_info_list = CONSTANTS.mcg, clustering_algorithm="spectral", no_q_computation=False, eigengap = False, silhouette=False, min_cluster_size=2, min_samples=1, plot_delta_heatmaps=True):
   
  aQ = []

  mss = []

  clusterings = []
  for prot_info in protein_info_list:
    #print(prot_info[0])

    if clustering_algorithm == "hdbscan":
      _,Q, ms = main_clustering(prot_info, k_cluster=prot_info[4], clustering_algorithm=clustering_algorithm, no_q_computation=no_q_computation, eigengap = eigengap, silhouette=silhouette, min_cluster_size=min_cluster_size, min_samples=min_samples, plot_delta_heatmaps=plot_delta_heatmaps)
      mss.append(ms)
    else:
      _,Q, ms = main_clustering(prot_info, k_cluster=prot_info[4], clustering_algorithm=clustering_algorithm, no_q_computation=no_q_computation, eigengap = eigengap, silhouette=silhouette, min_cluster_size=min_cluster_size, min_samples=min_samples, plot_delta_heatmaps=plot_delta_heatmaps)
      mss.append(ms)
    aQ.append(Q)
  
    clusterings.append(_)
  print('mean and std: ')
  print(str(np.mean(np.array(aQ))))
  print(str(np.std(np.array(aQ))))

  return clusterings, aQ, mss


def run_clustering_get_delta(protein_info_list = CONSTANTS.mcg, clustering_algorithm="spectral", no_q_computation=False, eigengap = False, silhouette=False, min_cluster_size=2, min_samples=1, plot_delta_heatmaps=True):
   
  aQ = []

  mss = []
  chunk_delta = False
  clusterings = []

  k_cluster=4
  for prot_info in protein_info_list:
    #print(prot_info[0])

    if clustering_algorithm == "hdbscan":
      #deltam = get_delta_matrices(prot_info, k_cluster=prot_info[4], clustering_algorithm=clustering_algorithm, no_q_computation=no_q_computation, eigengap = eigengap, silhouette=silhouette, min_cluster_size=min_cluster_size, min_samples=min_samples, plot_delta_heatmaps=plot_delta_heatmaps)
      frames_count = prot_info[3]
      traj_array, k_cluster = preprocessing(prot_info,frames_count,k_cluster)
      
      dist_matrices = get_distance_matrices(traj_array)
      del traj_array
      if chunk_delta:
        delta_matrices = get_delta_matrices(dist_matrices)
      else:
        delta_matrices = get_delta_matrices_no_chunk(dist_matrices)
      
      mss.append(delta_matrices)
    else:
      frames_count = prot_info[3]
      traj_array, k_cluster = preprocessing(prot_info,frames_count,k_cluster)
      
      dist_matrices = get_distance_matrices(traj_array)
      del traj_array
      if chunk_delta:
        delta_matrices = get_delta_matrices(dist_matrices)
      else:
        delta_matrices = get_delta_matrices_no_chunk(dist_matrices)
      #deltam = get_delta_matrices(prot_info, k_cluster=prot_info[4], clustering_algorithm=clustering_algorithm, no_q_computation=no_q_computation, eigengap = eigengap, silhouette=silhouette, min_cluster_size=min_cluster_size, min_samples=min_samples, plot_delta_heatmaps=plot_delta_heatmaps)
      mss.append(delta_matrices)
    #aQ.append(Q)
  
  #  clusterings.append(_)
  #print('mean and std: ')
  #print(str(np.mean(np.array(aQ))))
  #print(str(np.std(np.array(aQ))))

  return mss



def run_clustering_var(protein_info_list = CONSTANTS.mcg, clustering_algorithm="spectral_pyrea", no_q_computation=False, eigengap = False, silhouette=False, min_cluster_size=2, min_samples=1, fuser_alg="agreement", plot_delta_heatmaps=True):
   
  aQ = []

  mss = []

  clusterings = []
  for prot_info in protein_info_list:
    #print(prot_info[0])

    if clustering_algorithm == "spectral_pyrea":
      #_,Q, ms = main_clustering_varadd(prot_info, k_cluster=prot_info[4], clustering_algorithm=clustering_algorithm, no_q_computation=no_q_computation, fuser_alg=fuser_alg)
      _,Q = main_clustering_varadd(prot_info, k_cluster=prot_info[4], no_q_computation=no_q_computation, fuser_alg=fuser_alg, silhouette=silhouette, plot_delta_heatmaps=plot_delta_heatmaps)

    
    aQ.append(Q)
  
    clusterings.append(_)
  print('mean and std: ')
  print(str(np.mean(np.array(aQ))))
  print(str(np.std(np.array(aQ))))

  return clusterings, aQ



