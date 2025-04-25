import math
import CONSTANTS
import mdtraj as md
import distance_matrix as dm
import numpy as np
import multiprocessing
import warnings
import compare_clusterings as cc

from functools import wraps
from time import time



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

from functools import partial

'''
@timing
def get_distance_matrices(traj_array):
    return np.array(list(pool.map(dm.calculate_distance_matrix, traj_array)))
'''

@timing
def get_distance_matrices(traj_array, metric='euclidean'):
    # Use functools.partial to pass 'metric' argument to the function
    partial_distance_func = partial(dm.calculate_distance_matrix, metric=metric)
    return np.array(list(pool.map(partial_distance_func, traj_array)))

@timing
def get_delta_matrices(dist_matrices):
    return np.absolute(np.diff(dist_matrices,axis=0))

@timing
def get_delta_matrices_wo_absolute(dist_matrices):
    return np.diff(dist_matrices,axis=0)

@timing
def calculate_average_delta_matrix(delta_matrices):
    return np.sum(delta_matrices, axis=0)/len(delta_matrices)


def get_stddv(dist_matrices):
    L = dist_matrices.shape[0]
    mean_d = np.mean(dist_matrices,axis=0)
    fluctuations = (dist_matrices - mean_d)**2
    mean_fluctuations = np.mean(fluctuations, axis=0)

    S = np.sqrt((L / (L - 1)) * mean_fluctuations) 

    return S


@timing
def calculate_median_delta_matrix(delta_matrices):
    return np.median(delta_matrices, axis=0)


#new
@timing
def get_var_matrices(dist_matrices):
    return np.var(dist_matrices,axis=0)


@timing
def get_std_matrices(dist_matrices):
    return np.std(dist_matrices,axis=0)

#new, not used
@timing
def calculate_var_delta_matrix(delta_matrices):
    return np.var(delta_matrices, axis=0)/len(delta_matrices)


def prepare_data_from_df(x, use_mean_preprocessing=True, group_by_obj_id=False):
    import pandas as pd
    import numpy as np  

    if use_mean_preprocessing:
        # Aggregate duplicates by taking the mean of coordinates
        x = x.groupby(['t', 'obj_id'], as_index=False).mean()

    tpoints = []
    df_points = []
    traj_array = []
    point_array = []

    # Group by time ('t') after ensuring data is sorted by 'obj_id'

    if group_by_obj_id:
        group_variable = "obj_id"
        sort_variable = "t"
    else:
        group_variable = "t"
        sort_variable = "obj_id"

    for g in x.sort_values([sort_variable], ascending=True).groupby(group_variable):
        tpoints.append(g[1].values)
        df_points.append(pd.DataFrame(g[1]))
        new_df = pd.DataFrame(g[1])

        # Extract trajectory data (x, y, z) and object IDs
        traj_array.append(np.array(new_df[['x', 'y', 'z']].values))
        point_array.append(new_df[['obj_id']].values)

    # Calculate the number of frames and objects
    frames_count = len(df_points[0]) if df_points else 0
    n_objects = len(df_points)

    return traj_array, point_array, frames_count, n_objects


def prepare_data_from_df_ndim(x, use_mean_preprocessing=True, group_by_obj_id=False):
    import pandas as pd
    import numpy as np  

    if use_mean_preprocessing:
        x = x.groupby(['t', 'obj_id'], as_index=False).mean()

    tpoints = []
    df_points = []
    traj_array = []
    point_array = []

    # Detect spatial coordinate columns (e.g. x1, x2, ..., xn)
    coord_cols = [col for col in x.columns if col.startswith('x') and col[1:].isdigit()]
    coord_cols.sort(key=lambda c: int(c[1:]))  # Ensure x1, x2, ..., xn order

    # Grouping logic
    group_variable = "obj_id" if group_by_obj_id else "t"
    sort_variable = "t" if group_by_obj_id else "obj_id"

    for _, group_df in x.sort_values([sort_variable], ascending=True).groupby(group_variable):
        tpoints.append(group_df.values)
        df_points.append(pd.DataFrame(group_df))
        traj_array.append(group_df[coord_cols].values)
        point_array.append(group_df[['obj_id']].values)

    frames_count = len(df_points[0]) if df_points else 0
    n_objects = len(df_points)

    return traj_array, point_array, frames_count, n_objects

def mean_preprocess_dataframe_onlymean(df):
    """
    Groups the DataFrame by 't' and 'obj_id', computes the mean for numeric columns, 
    and returns the processed DataFrame.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame containing 't', 'obj_id', and numeric columns.

    Returns:
        pd.DataFrame: Processed DataFrame with mean values for duplicates.
    """
    import pandas as pd

    # Group by 't' and 'obj_id' and compute the mean for numeric columns
    processed_df = df.groupby(['t', 'obj_id'], as_index=False).mean()

    return processed_df

def mean_preprocess_dataframe_cakmak_onlymean(df):
    """
    Groups the DataFrame by 't' and 'obj_id', computes the mean for numeric columns, 
    and returns the processed DataFrame.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame containing 't', 'obj_id', and numeric columns.

    Returns:
        pd.DataFrame: Processed DataFrame with mean values for duplicates.
    """
    import pandas as pd

    # Group by 't' and 'obj_id' and compute the mean for numeric columns
    processed_df = df.groupby(['frame', 'id'], as_index=False).mean()

    return processed_df


# versions with majority voting
import pandas as pd

def mean_preprocess_dataframe(df):
    """
    Groups the DataFrame by 't' and 'obj_id', computes the mean for numeric columns, 
    and assigns the most frequent 'label' based on 'obj_id' across all t values.
    
    Ensures that 't' and 'obj_id' are integers in the final DataFrame.

    Parameters:
        df (pd.DataFrame): Input DataFrame containing 't', 'obj_id', 'label', and numeric columns.

    Returns:
        pd.DataFrame: Processed DataFrame with mean values for numeric columns 
                      and majority-voted 'label' per obj_id.
    """
    # Ensure 't' and 'obj_id' are treated as integers in input
    df['t'] = df['t'].astype(int)
    df['obj_id'] = df['obj_id'].astype(int)

    # Exclude 'label' from mean computation
    numeric_cols = df.select_dtypes(include=['number']).columns.difference(['label'])
    
    # Compute the mean for numeric columns grouped by 't' and 'obj_id'
    processed_df = df.groupby(['t', 'obj_id'], as_index=False)[numeric_cols].mean()

    # Ensure 't' and 'obj_id' remain integers after aggregation
    processed_df['t'] = processed_df['t'].astype(int)
    processed_df['obj_id'] = processed_df['obj_id'].astype(int)

    # Determine the most frequent 'label' for each 'obj_id' (ignoring 't')
    def majority_vote(series):
        return series.value_counts().idxmax()  # Selects the most frequent label per obj_id

    label_df = df.groupby('obj_id')['label'].agg(majority_vote).reset_index()

    # Merge the mean-computed DataFrame with the majority-voted label DataFrame
    processed_df = processed_df.merge(label_df, on='obj_id', how='left')

    return processed_df


def mean_preprocess_dataframe_cakmak(df):
    """
    Groups the DataFrame by 't' and 'obj_id', computes the mean for numeric columns, 
    and assigns the most frequent 'label' based on 'obj_id' across all t values.
    
    Ensures that 't' and 'obj_id' are integers in the final DataFrame.

    Parameters:
        df (pd.DataFrame): Input DataFrame containing 't', 'obj_id', 'label', and numeric columns.

    Returns:
        pd.DataFrame: Processed DataFrame with mean values for numeric columns 
                      and majority-voted 'label' per obj_id.
    """
    # Ensure 't' and 'obj_id' are treated as integers in input
    df['frame'] = df['frame'].astype(int)
    df['id'] = df['id'].astype(int)

    # Exclude 'label' from mean computation
    numeric_cols = df.select_dtypes(include=['number']).columns.difference(['cid'])
    
    # Compute the mean for numeric columns grouped by 't' and 'obj_id'
    processed_df = df.groupby(['frame', 'id'], as_index=False)[numeric_cols].mean()

    # Ensure 't' and 'obj_id' remain integers after aggregation
    processed_df['frame'] = processed_df['frame'].astype(int)
    processed_df['id'] = processed_df['id'].astype(int)

    # Determine the most frequent 'label' for each 'obj_id' (ignoring 't')
    def majority_vote(series):
        return series.value_counts().idxmax()  # Selects the most frequent label per obj_id

    label_df = df.groupby('id')['cid'].agg(majority_vote).reset_index()

    # Merge the mean-computed DataFrame with the majority-voted label DataFrame
    processed_df = processed_df.merge(label_df, on='id', how='left')

    return processed_df




def main_clustering_general_get_dist_delta(x,k_cluster=None,assign_labels='discretize', clustering_algorithm="spectral", no_q_computation=False, eigengap = False, silhouette=False, min_cluster_size=2, min_samples=1, plot_delta_heatmaps=True) -> np.ndarray:
  
  import pandas as pd 
  tpoints = []
  df_points = []
  traj_array = []
  point_array = []
  #for g in x.sort_values(['t'],ascending=True).groupby("obj_id"):
  for g in x.sort_values(['obj_id'],ascending=True).groupby("t"):
      tpoints.append(g[1].values)
      df_points.append(pd.DataFrame(g[1]))
      new_df = pd.DataFrame(g[1])
      traj_array.append(np.array(new_df[['x', 'y', 'z']].values))
      point_array.append(new_df[['obj_id']].values)

  frames_count = len(df_points[0])
  n_objects = len(df_points)
  #traj_array, k_cluster = preprocessing(prot_info,frames_count,k_cluster)


  if not k_cluster:
    k_cluster = int(math.sqrt(n_objects))
  dist_matrices = get_distance_matrices(traj_array)
  delta_matrices = get_delta_matrices(dist_matrices)

  return dist_matrices, delta_matrices


def main_clustering_general(x,k_cluster=None,assign_labels='discretize', clustering_algorithm="spectral", no_q_computation=False, eigengap = False, silhouette=False, min_cluster_size=2, min_samples=1, plot_delta_heatmaps=True) -> np.ndarray:
  
  import pandas as pd 
  tpoints = []
  df_points = []
  traj_array = []
  point_array = []
  #for g in x.sort_values(['t'],ascending=True).groupby("obj_id"):
  for g in x.sort_values(['obj_id'],ascending=True).groupby("t"):
      tpoints.append(g[1].values)
      df_points.append(pd.DataFrame(g[1]))
      new_df = pd.DataFrame(g[1])
      if "z" in new_df:
        traj_array.append(np.array(new_df[['x', 'y', 'z']].values))
      else:
        traj_array.append(np.array(new_df[['x', 'y']].values))
      point_array.append(new_df[['obj_id']].values)

  frames_count = len(df_points[0])
  n_objects = len(df_points)
  #traj_array, k_cluster = preprocessing(prot_info,frames_count,k_cluster)


  if not k_cluster:
    k_cluster = int(math.sqrt(n_objects))
  dist_matrices = get_distance_matrices(traj_array)
  delta_matrices = get_delta_matrices(dist_matrices)
  average_delta_matrix = calculate_average_delta_matrix(delta_matrices)

  matrix = []

  if clustering_algorithm == "spectral":
    clustering = dm.spectral_clustering_on_deltas(average_delta_matrix, k_cluster, assign_labels, eigengap = eigengap, silhouette = silhouette, plot_delta_heatmaps=plot_delta_heatmaps)
  elif clustering_algorithm == "hdbscan":
     clustering, k_cluster, matrix = dm.hdbscan_clustering_on_deltas(average_delta_matrix, min_cluster_size=min_cluster_size, min_samples=min_samples)
  elif clustering_algorithm == "agglomerative":
     clustering, k_cluster, matrix = dm.agglomerative_clustering_on_deltas_ward(average_delta_matrix, None)


  Q = []
  

  if not no_q_computation: 
    try:
      Q, _ = cc.get_Q_for_clustering(dist_matrices, clustering, k_cluster)
    except:
      print("something wrong with clustering Q computation")


  return clustering, Q, matrix, point_array



def main_clustering_general_stddv(x,k_cluster=None,assign_labels='discretize', clustering_algorithm="spectral", no_q_computation=False, eigengap = False, silhouette=False, min_cluster_size=2, min_samples=1, plot_delta_heatmaps=True) -> np.ndarray:
  
  import pandas as pd 
  tpoints = []
  df_points = []
  traj_array = []
  point_array = []
  #for g in x.sort_values(['t'],ascending=True).groupby("obj_id"):
  for g in x.sort_values(['obj_id'],ascending=True).groupby("t"):
      tpoints.append(g[1].values)
      df_points.append(pd.DataFrame(g[1]))
      new_df = pd.DataFrame(g[1])
      if "z" in new_df:
        traj_array.append(np.array(new_df[['x', 'y', 'z']].values))
      else:
        traj_array.append(np.array(new_df[['x', 'y']].values))
      point_array.append(new_df[['obj_id']].values)

  frames_count = len(df_points[0])
  n_objects = len(df_points)
  #traj_array, k_cluster = preprocessing(prot_info,frames_count,k_cluster)


  if not k_cluster:
    k_cluster = int(math.sqrt(n_objects))
  dist_matrices = get_distance_matrices(traj_array)
  #delta_matrices = get_delta_matrices(dist_matrices)
  #average_delta_matrix = calculate_average_delta_matrix(delta_matrices)
  stddv_matrix = get_stddv(dist_matrices) 

  matrix = []

  if clustering_algorithm == "spectral":
    clustering = dm.spectral_clustering_on_deltas(stddv_matrix, k_cluster, assign_labels, eigengap = eigengap, silhouette = silhouette, plot_delta_heatmaps=plot_delta_heatmaps)
  elif clustering_algorithm == "hdbscan":
     clustering, k_cluster, matrix = dm.hdbscan_clustering_on_deltas(stddv_matrix, min_cluster_size=min_cluster_size, min_samples=min_samples)
  elif clustering_algorithm == "agglomerative":
     clustering, k_cluster, matrix = dm.agglomerative_clustering_on_deltas_ward(stddv_matrix, None)


  Q = []
  

  if not no_q_computation: 
    try:
      Q, _ = cc.get_Q_for_clustering(dist_matrices, clustering, k_cluster)
    except:
      print("something wrong with clustering Q computation")


  return clustering, Q, matrix, point_array




def main_clustering_general_std(x,k_cluster=None,assign_labels='discretize', clustering_algorithm="spectral", no_q_computation=False, eigengap = False, fuser_alg = "agreement", silhouette=False, min_cluster_size=2, min_samples=1, plot_delta_heatmaps=True) -> np.ndarray:
  #lets assume dataframe with t and x.y.z
  import pandas as pd 
  tpoints = []
  df_points = []
  traj_array = []
  point_array = []
  #for g in x.sort_values(['t'],ascending=True).groupby("obj_id"):
  for g in x.sort_values(['obj_id'],ascending=True).groupby("t"):

      tpoints.append(g[1].values)
      df_points.append(pd.DataFrame(g[1]))
      new_df = pd.DataFrame(g[1])
      traj_array.append(np.array(new_df[['x', 'y', 'z']].values))
      point_array.append(new_df[['obj_id']].values)

  frames_count = len(df_points[0])
  n_objects = len(df_points)
  #traj_array, k_cluster = preprocessing(prot_info,frames_count,k_cluster)


  if not k_cluster:
    k_cluster = int(math.sqrt(n_objects))
  dist_matrices = get_distance_matrices(traj_array)
  delta_matrices = get_delta_matrices(dist_matrices)
  average_delta_matrix = calculate_average_delta_matrix(delta_matrices)

  std_delta_matrix = get_std_matrices(dist_matrices)


  matrix = []


 
  clustering, k_cluster, matrix = dm.hdbscan_clustering_on_deltas_varadd(average_delta_matrix, std_delta_matrix, min_cluster_size=min_cluster_size, min_samples=min_samples)


  Q = []
  

  if not no_q_computation: 
    try:
      Q, _ = cc.get_Q_for_clustering(dist_matrices, clustering, k_cluster)
    except:
      print("something wrong with clustering Q computation")


  return clustering, Q, matrix, point_array






def main_clustering_general_std_distance(x,k_cluster=None,assign_labels='discretize', clustering_algorithm="spectral", no_q_computation=False, eigengap = False, fuser_alg = "agreement", silhouette=False, min_cluster_size=2, min_samples=1, plot_delta_heatmaps=True) -> np.ndarray:
  import pandas as pd 
  tpoints = []
  df_points = []
  traj_array = []
  point_array = []
  #for g in x.sort_values(['t'],ascending=True).groupby("obj_id"):
  for g in x.sort_values(['obj_id'],ascending=True).groupby("t"):

      tpoints.append(g[1].values)
      df_points.append(pd.DataFrame(g[1]))
      new_df = pd.DataFrame(g[1])
      traj_array.append(np.array(new_df[['x', 'y', 'z']].values))
      point_array.append(new_df[['obj_id']].values)

  frames_count = len(df_points[0])
  n_objects = len(df_points)
  #traj_array, k_cluster = preprocessing(prot_info,frames_count,k_cluster)


  if not k_cluster:
    k_cluster = int(math.sqrt(n_objects))
  dist_matrices = get_distance_matrices(traj_array)
  delta_matrices = get_delta_matrices(dist_matrices)
  average_delta_matrix = calculate_average_delta_matrix(delta_matrices)

  std_delta_matrix = get_std_matrices(dist_matrices)


  matrix = []


  clustering, k_cluster, matrix = dm.hdbscan_clustering_on_deltas_varadd_distance(average_delta_matrix, std_delta_matrix, min_cluster_size=min_cluster_size, min_samples=min_samples)


  Q = []
  

  if not no_q_computation: 
    try:
      Q, _ = cc.get_Q_for_clustering(dist_matrices, clustering, k_cluster)
    except:
      print("something wrong with clustering Q computation")


  return clustering, Q, matrix, point_array



