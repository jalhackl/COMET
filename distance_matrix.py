from ast import Num
from turtle import update
import numpy as np
import math
import mdtraj as md
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import scipy
from sklearn.cluster import SpectralClustering
from sklearn.cluster import AgglomerativeClustering
import sklearn.neighbors as skn
import CONSTANTS
#import manipulate_trajectory as mt

from functools import wraps, reduce
from time import time

def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print('%2.4f' % (te-ts))
        #print('func:%r took: %2.4f sec' % (f.__name__, te-ts))
        return result
    return wrap

def show_heatmaps_of_delta_matrices(traj_array: np.ndarray, frames):

  fig, ax = plt.subplots()
  fig.suptitle('Heatmap of Delta Matrices of Protein R1')

  def get_animation_for_frame(i):
    if i == 0:
      return
    prev_dist_matrix = calculate_distance_matrix(traj_array[frames[i-1]])
    dist_matrix = calculate_distance_matrix(traj_array[frames[i]])
    delta_matrix = calculate_delta_matrix(prev_dist_matrix, dist_matrix)
    imshow = plt.imshow(delta_matrix, cmap='viridis', interpolation='nearest', vmin=0, vmax=0.2)
    ax.set_title('Frame ' + str(frames[i]))
    if i == 1:
      fig.colorbar(imshow, ax=ax)
      ax.set_xlabel('Alpha Carbon Atom Indices')
      ax.set_ylabel('Alpha Carbon Atom Indices')

  ani = FuncAnimation(fig, get_animation_for_frame, frames=len(frames), interval=1000, repeat=False)
  ani.save(CONSTANTS.tmp_prefix + 'heatmap_r1.gif', writer='imagemagick', fps=1, dpi=250)
  #plt.show()

def sum_of_all_deltas(traj_arrays: np.ndarray) -> np.ndarray:
  '''
  Get the sum of all delta matrices
  '''
  atom_count = len(traj_arrays[0])
  sum_matrix = np.zeros((atom_count, atom_count))
  prev_dist_matrix = None

  for frame_array in traj_arrays:
    dist_matrix = calculate_distance_matrix(frame_array)
    # print('Min value of distance: ' + str(np.min(dist_matrix[np.nonzero(dist_matrix)])))
    if prev_dist_matrix is not None:
      delta_matrix = calculate_delta_matrix(prev_dist_matrix, dist_matrix)
      # print('Avg value: ' + str(calculate_average_value(delta_matrix)))
      sum_matrix += delta_matrix
    prev_dist_matrix = dist_matrix

  return sum_matrix


def create_delta_heatmap(distance_matrix, title="avg. delta matrix heatmap", label = "Object Index"):
    import seaborn as sns
    plt.figure(figsize=(10, 8))
    sns.heatmap(distance_matrix, annot=False, cmap="YlGnBu", fmt=".1f")
    plt.title(title)
    plt.xlabel(label)
    plt.ylabel(label)
    plt.show()


def clustering_on_deltas(summed_delta_matrix: np.ndarray, algorithm: str, plot_heatmap=False, title_info = "avg. delta matrix heatmap", **kwargs) -> np.ndarray:
    """
    General function to apply different clustering algorithms on the given delta matrix.

    Parameters:
    - summed_delta_matrix (np.ndarray): The input matrix representing pairwise dissimilarities.
    - algorithm (str): The clustering algorithm to use ('spectral', 'affinity', 'kmeans', etc.).
    - plot_delta_heatmaps (bool, default=True): Whether to plot the heatmap of the delta matrix.
    - **kwargs: Additional parameters specific to the chosen clustering algorithm.

    Returns:
    - np.ndarray: Cluster labels for each data point.
    """

    if plot_heatmap:
        create_delta_heatmap(summed_delta_matrix, title=f"{algorithm} clustering - f{title_info}")
    
    import inspect    
    # Dispatch the clustering algorithm
    if algorithm.lower() == "spectral":
        # Inspect Spectral Clustering parameters
        valid_params = inspect.signature(spectral_clustering_on_deltas).parameters
        # Filter kwargs based on valid parameters
        params = {key: value for key, value in kwargs.items() if key in valid_params}
        return spectral_clustering_on_deltas(summed_delta_matrix, **params)
    elif algorithm.lower() == "affinity":
        valid_params = inspect.signature(affinity_propagation_on_deltas).parameters
        # Filter kwargs based on valid parameters
        params = {key: value for key, value in kwargs.items() if key in valid_params}
        return affinity_propagation_on_deltas(summed_delta_matrix, **params)
    elif algorithm.lower() == "hdbscan":
        valid_params = inspect.signature(hdbscan_clustering_on_deltas).parameters
        # Filter kwargs based on valid parameters
        params = {key: value for key, value in kwargs.items() if key in valid_params}
        return hdbscan_clustering_on_deltas(summed_delta_matrix, **params)
    elif algorithm.lower() == "agglomerative":
        valid_params = inspect.signature(agglomerative_clustering_on_deltas).parameters
        # Filter kwargs based on valid parameters
        params = {key: value for key, value in kwargs.items() if key in valid_params}
        return agglomerative_clustering_on_deltas(summed_delta_matrix, **params)
    elif algorithm.lower() == "multispectral":
        valid_params = inspect.signature(apply_pyrea).parameters
        # Filter kwargs based on valid parameters
        params = {key: value for key, value in kwargs.items() if key in valid_params}
        return apply_pyrea(summed_delta_matrix, **params)
    else:
        raise ValueError(f"Unknown clustering algorithm: {algorithm}")



@timing
def spectral_clustering_on_deltas(summed_delta_matrix: np.ndarray, cluster_count: Num, assign_labels='discretize', eigengap=False, silhouette = False, plot_delta_heatmaps=False) -> np.ndarray:

  if plot_delta_heatmaps:
     create_delta_heatmap(summed_delta_matrix, title="avg. delta matrix heatmap")
     

  normed_similarity_matrix = 1 - normed_values(summed_delta_matrix)


  if eigengap == True:
    cluster_count= predict_k(normed_similarity_matrix)

  if silhouette == True:
    cluster_count = optimal_number_of_clusters_silhouette(summed_delta_matrix)


  clustering = SpectralClustering(n_clusters=cluster_count, assign_labels=assign_labels, affinity='precomputed')



  result = clustering.fit(normed_similarity_matrix)
  return result.labels_


@timing
def affinity_propagation_on_deltas(summed_delta_matrix: np.ndarray, damping=0.5, preference=None, plot_delta_heatmaps=False) -> np.ndarray:
    """
    Applies Affinity Propagation clustering on the given delta matrix.

    Parameters:
    - summed_delta_matrix (np.ndarray): The input matrix representing pairwise dissimilarities.
    - damping (float, default=0.5): Damping factor for Affinity Propagation (range: 0.5 to 1).
    - preference (float or None, default=None): Preferences for data points to be chosen as exemplars.
    - plot_delta_heatmaps (bool, default=True): Whether to plot the heatmap of the delta matrix.

    Returns:
    - np.ndarray: Cluster labels for each data point.
    """
    from sklearn.cluster import AffinityPropagation
    if plot_delta_heatmaps:
        create_delta_heatmap(summed_delta_matrix, title="avg. delta matrix heatmap")

    
    normed_similarity_matrix = 1 - normed_values(summed_delta_matrix)

    # Apply Affinity Propagation
    clustering = AffinityPropagation(damping=damping, preference=preference, affinity='precomputed')
    result = clustering.fit(normed_similarity_matrix)

    return result.labels_



@timing
def hdbscan_clustering_on_deltas(summed_delta_matrix: np.ndarray, min_cluster_size=2, min_samples=1, return_matrix = True, plot_delta_heatmaps=False) -> np.ndarray:
  if plot_delta_heatmaps:
     create_delta_heatmap(summed_delta_matrix, title="avg. delta matrix heatmap")
     

  normed_similarity_matrix = normed_values(summed_delta_matrix)


  import hdbscan

  clustering = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, metric='precomputed') 


  result = clustering.fit(normed_similarity_matrix.astype(np.float64))

  if return_matrix == False:
    return result.labels_, result.labels_.max()
  else:
    return result.labels_, result.labels_.max(), normed_similarity_matrix



@timing
def hdbscan_clustering_on_deltas_varadd(summed_delta_matrix: np.ndarray, std_delta_matrix: np.ndarray, min_cluster_size=2, min_samples=1, return_matrix = True, plot_delta_heatmaps=False) -> np.ndarray:
  normed_similarity_matrix = 1 - normed_values(summed_delta_matrix)


  summed_delta_matrix_1std = summed_delta_matrix + std_delta_matrix 
  summed_delta_matrix_2std = summed_delta_matrix + std_delta_matrix * 2

  if plot_delta_heatmaps:
    create_delta_heatmap(summed_delta_matrix, title="avg. delta matrix heatmap")
    create_delta_heatmap(summed_delta_matrix_1std, title="avg. delta matrix heatmap + 1STD")
    create_delta_heatmap(summed_delta_matrix_2std, title="avg. delta matrix heatmap + 2STD")

  normed_similarity_matrix_1std = 1 - normed_values(summed_delta_matrix_1std)
  normed_similarity_matrix_2std = 1 - normed_values(summed_delta_matrix_2std)


  import hdbscan

  clustering = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, metric='precomputed') 


  result = clustering.fit(normed_similarity_matrix_2std.astype(np.float64))

  if return_matrix == False:
    return result.labels_, result.labels_.max()
  else:
    return result.labels_, result.labels_.max(), normed_similarity_matrix_2std
  

@timing
def hdbscan_clustering_on_deltas_varadd_distance(summed_delta_matrix: np.ndarray, std_delta_matrix: np.ndarray, min_cluster_size=2, min_samples=1, return_matrix = True, plot_delta_heatmaps=False) -> np.ndarray:
  normed_similarity_matrix = 1 - normed_values(summed_delta_matrix)


  summed_delta_matrix_1std = summed_delta_matrix + std_delta_matrix 
  summed_delta_matrix_2std = summed_delta_matrix + std_delta_matrix * 2

  if plot_delta_heatmaps:
    create_delta_heatmap(summed_delta_matrix, title="avg. delta matrix heatmap")
    create_delta_heatmap(summed_delta_matrix_1std, title="avg. delta matrix heatmap + 1STD")
    create_delta_heatmap(summed_delta_matrix_2std, title="avg. delta matrix heatmap + 2STD")

  normed_similarity_matrix_1std = 1 - normed_values(summed_delta_matrix_1std)
  normed_similarity_matrix_2std = 1 - normed_values(summed_delta_matrix_2std)

  if plot_delta_heatmaps:
    create_delta_heatmap(normed_similarity_matrix, title="avg. delta-dist matrix heatmap")
    create_delta_heatmap(normed_similarity_matrix_1std, title="avg. delta-dist matrix heatmap + 1STD")
    create_delta_heatmap(normed_similarity_matrix_2std, title="avg. delta-dist matrix heatmap + 2STD")

  import hdbscan

  clustering = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, metric='precomputed') 


  result = clustering.fit(summed_delta_matrix_2std.astype(np.float64))


  if return_matrix == False:
    return result.labels_, result.labels_.max()
  else:
    return result.labels_, result.labels_.max(), normed_similarity_matrix_2std
  


def spectral_clustering_on_deltas_varadd(summed_delta_matrix: np.ndarray, std_delta_matrix: np.ndarray,cluster_count: Num, assign_labels='discretize', fuser_alg = "agreement", silhouette = False, plot_delta_heatmaps=False) -> np.ndarray:
  #from mvlearn.cluster import MultiviewSpectralClustering
  import pyrea

  normed_similarity_matrix = 1 - normed_values(summed_delta_matrix)


  summed_delta_matrix_1std = summed_delta_matrix + std_delta_matrix 
  summed_delta_matrix_2std = summed_delta_matrix + std_delta_matrix * 2

  if plot_delta_heatmaps:
    create_delta_heatmap(summed_delta_matrix, title="avg. delta matrix heatmap")
    create_delta_heatmap(summed_delta_matrix_1std, title="avg. delta matrix heatmap + 1STD")
    create_delta_heatmap(summed_delta_matrix_2std, title="avg. delta matrix heatmap + 2STD")

  normed_similarity_matrix_1std = 1 - normed_values(summed_delta_matrix_1std)
  normed_similarity_matrix_2std = 1 - normed_values(summed_delta_matrix_2std)



  if silhouette == True:
    cluster_count = optimal_number_of_clusters_silhouette(summed_delta_matrix, std_delta_matrix, cluster_alg="spectral_pyrea", assign_labels='discretize', fuser_alg = "agreement")


  f = pyrea.fuser(fuser_alg)
  #f = pyrea.fuser('consensus')

  hc1 = pyrea.clusterer('spectral',  n_clusters=cluster_count, assign_labels=assign_labels, affinity="precomputed")
  hc2 = pyrea.clusterer('spectral',  n_clusters=cluster_count, assign_labels=assign_labels, affinity="precomputed")

  v1 = pyrea.view(normed_similarity_matrix, hc1)
  v2 = pyrea.view(normed_similarity_matrix_1std, hc1)
  v3 = pyrea.view(normed_similarity_matrix_2std, hc1)

  #v2 = pyrea.view(normed_var_matrix, hc1)

  hc1_pre = pyrea.clusterer('spectral', n_clusters=cluster_count, assign_labels=assign_labels)
  v_ensemble_1 = pyrea.view(pyrea.execute_ensemble([v1, v2, v3], f), hc1_pre)

  result = v_ensemble_1.execute()
  #s_clusters_v1 = s_spectral.fit_predict(m_data[0])
  #s_clusters_v2 = s_spectral.fit_predict(m_data[1])
  return result


def apply_pyrea(normed_similarity_matrix, summed_delta_matrix_1std, summed_delta_matrix_2std, cluster_count, assign_labels="discretize", fuser_alg="agreement"):
    import pyrea

    f = pyrea.fuser(fuser_alg)
    #f = pyrea.fuser('consensus')


    hc1 = pyrea.clusterer('spectral',  n_clusters=cluster_count, assign_labels=assign_labels, affinity="precomputed")

    v1 = pyrea.view(normed_similarity_matrix, hc1)
    v2 = pyrea.view(summed_delta_matrix_1std, hc1)
    v3 = pyrea.view(summed_delta_matrix_2std, hc1)

    #v2 = pyrea.view(normed_var_matrix, hc1)

    hc1_pre = pyrea.clusterer('spectral', n_clusters=cluster_count, assign_labels=assign_labels)
    v_ensemble_1 = pyrea.view(pyrea.execute_ensemble([v1, v2, v3], f), hc1_pre)

    result = v_ensemble_1.execute()
    #s_clusters_v1 = s_spectral.fit_predict(m_data[0])
    #s_clusters_v2 = s_spectral.fit_predict(m_data[1])

    return result
   


@timing
def spectral_clustering_on_deltas_var(summed_delta_matrix: np.ndarray, var_delta_matrix: np.ndarray, cluster_count: Num, assign_labels='discretize', eigengap=False, silhouette=False, stdev_addition=True) -> np.ndarray:
  
  if stdev_addition:
    summed_delta_matrix = summed_delta_matrix + var_delta_matrix * 2
  
  normed_similarity_matrix = 1 - normed_values(summed_delta_matrix)



  if eigengap == True:
    cluster_count= predict_k(normed_similarity_matrix)


  if silhouette == True:
     cluster_count = optimal_number_of_clusters_silhouette(summed_delta_matrix)

  clustering = SpectralClustering(n_clusters=cluster_count, assign_labels=assign_labels, affinity='precomputed')


  result = clustering.fit(normed_similarity_matrix)
  return result.labels_



def optimal_number_of_clusters_silhouette(summed_delta_matrix, std_delta_matrix=None, max_clusters=None, plot_silhouette=True, plot_all_sihluette_samples=False, cluster_alg = "spectral", assign_labels='discretize', fuser_alg = "agreement"):
    from sklearn.metrics import silhouette_score, silhouette_samples
    from sklearn.cluster import SpectralClustering

    #clusters with an avg length < 4 are rather dubious...
    if not max_clusters:
       max_clusters = int(len(summed_delta_matrix) * 0.25)

    silhouette_avg_list = []
    silhouette_scores_list = []
    
    for k in range(2, max_clusters+1):  
        
        if cluster_alg == "spectral":
          normed_similarity_matrix = 1 - normed_values(summed_delta_matrix)
          
          spectral_model = SpectralClustering(n_clusters=k, assign_labels=assign_labels, affinity='precomputed')
          clusters = spectral_model.fit_predict(normed_similarity_matrix)
        elif cluster_alg == "spectral_pyrea":
          normed_similarity_matrix = 1 - normed_values(summed_delta_matrix)


          summed_delta_matrix_1std = summed_delta_matrix + std_delta_matrix 
          summed_delta_matrix_2std = summed_delta_matrix + std_delta_matrix * 2

          normed_similarity_matrix_1std = 1 - normed_values(summed_delta_matrix_1std)
          normed_similarity_matrix_2std = 1 - normed_values(summed_delta_matrix_2std)
          clusters = apply_pyrea(normed_similarity_matrix,normed_similarity_matrix_1std, normed_similarity_matrix_2std, k, assign_labels=assign_labels, fuser_alg=fuser_alg)

        
        silhouette_scores = silhouette_samples(summed_delta_matrix, clusters, metric = "precomputed")
        silhouette_scores_list.append(silhouette_scores)
        
        silhouette_avg = silhouette_score(summed_delta_matrix, clusters, metric = "precomputed")
        silhouette_avg_list.append(silhouette_avg)

        # Find the index where silhouette score is maximized
    optimal_k = np.argmax(silhouette_avg_list) + 2  
    

    if plot_all_sihluette_samples:
      for i, (silhouette_scores, k) in enumerate(zip(silhouette_scores_list, range(2, max_clusters+1)), 1):
          y_lower = 10
          for cluster in range(k):
              cluster_silhouette_scores = silhouette_scores[clusters == cluster]
              cluster_silhouette_scores.sort()
              size_cluster_i = cluster_silhouette_scores.shape[0]
              y_upper = y_lower + size_cluster_i
              plt.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_silhouette_scores, alpha=0.7)
              plt.text(-0.05, y_lower + 0.5 * size_cluster_i, str(cluster))
              y_lower = y_upper + 10
          plt.axvline(x=silhouette_avg_list[i-1], color="red", linestyle="--")
          plt.title("Silhouette plot for {} clusters".format(k))
          plt.xlabel("Silhouette coefficient")
          plt.ylabel("Cluster label")
          plt.tight_layout()

          plt.show()

    else:
      plt.figure(figsize=(8, 6))

      silhouette_scores = silhouette_scores_list[optimal_k - 2]
      y_lower = 10
      for cluster in range(optimal_k):
        cluster_silhouette_scores = silhouette_scores[clusters == cluster]


        cluster_silhouette_scores.sort()
        size_cluster_i = cluster_silhouette_scores.shape[0]
        y_upper = y_lower + size_cluster_i
        plt.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_silhouette_scores, alpha=0.7)
        plt.text(-0.05, y_lower + 0.5 * size_cluster_i, str(cluster))
        y_lower = y_upper + 10
      plt.axvline(x=silhouette_avg_list[optimal_k-2], color="red", linestyle="--")
      plt.title("Silhouette plot for {} clusters".format(optimal_k))
      plt.xlabel("Silhouette coefficient")
      plt.ylabel("Cluster label")
      plt.tight_layout()

      plt.show()

    
    plt.show()


    # Plot silhouette scores
    if plot_silhouette:
        plt.plot(range(2, max_clusters+1), silhouette_avg_list, marker='o')
        plt.xlabel('Number of Clusters')
        plt.ylabel('Silhouette Score')
        plt.title('Silhouette Score for Different Numbers of Clusters')
        plt.show()
      

    return optimal_k #, silhouette_avg_list



@timing
def hdbscan_clustering_on_deltas_var(summed_delta_matrix: np.ndarray, var_delta_matrix = None, cluster_count = None, assign_labels='discretize', min_cluster_size=2, min_samples=1, return_matrix = True, stdev_addition=True) -> np.ndarray:
  if stdev_addition:
    summed_delta_matrix = summed_delta_matrix + var_delta_matrix * 2

  normed_similarity_matrix = normed_values(summed_delta_matrix)


  import hdbscan

  clustering = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, metric='precomputed') 


  result = clustering.fit(normed_similarity_matrix.astype(np.float64))

  if return_matrix == False:
    return result.labels_, result.labels_.max()
  else:
    return result.labels_, result.labels_.max(), normed_similarity_matrix



def spectral_clustering_on_deltas_var(summed_delta_matrix: np.ndarray, var_delta_matrix: np.ndarray,cluster_count: Num, assign_labels='discretize') -> np.ndarray:
  from mvlearn.cluster import MultiviewSpectralClustering
  import pyrea

  normed_similarity_matrix = 1 - normed_values(summed_delta_matrix)

  normed_var_matrix = 1 - normed_values(var_delta_matrix)


  '''
  clustering = MultiviewSpectralClustering(n_clusters=cluster_count,
            affinity='precomputed', random_state=0)
  result = clustering.fit_predict(normed_similarity_matrix)
  '''
  f = pyrea.fuser('agreement')
  #f = pyrea.fuser('consensus')


  hc1 = pyrea.clusterer('spectral',  n_clusters=cluster_count, assign_labels=assign_labels, affinity="precomputed")
  hc2 = pyrea.clusterer('spectral',  n_clusters=cluster_count, assign_labels=assign_labels, affinity="precomputed")

  v1 = pyrea.view(normed_similarity_matrix, hc1)
  v2 = pyrea.view(normed_var_matrix, hc1)

  hc1_pre = pyrea.clusterer('spectral', n_clusters=cluster_count, assign_labels=assign_labels)
  v_ensemble_1 = pyrea.view(pyrea.execute_ensemble([v1, v2], f), hc1_pre)

  result = v_ensemble_1.execute()
  #s_clusters_v1 = s_spectral.fit_predict(m_data[0])
  #s_clusters_v2 = s_spectral.fit_predict(m_data[1])
  return result



from scipy.sparse.linalg import eigsh
from sklearn.manifold._spectral_embedding import _set_diag
from scipy.sparse import csgraph


#This function is taken from https://github.com/mingmingyang/auto_spectral_clustering/tree/master
def predict_k(affinity_matrix):
    """
    Predict number of clusters based on the eigengap.

    Parameters
    ----------
    affinity_matrix : array-like or sparse matrix, shape: (n_samples, n_samples)
        adjacency matrix.
        Each element of this matrix contains a measure of similarity between two of the data points.

    Returns
    ----------
    k : integer
        estimated number of cluster.

    Note
    ---------
    If graph is not fully connected, zero component as single cluster.

    References
    ----------
    A Tutorial on Spectral Clustering, 2007
        Luxburg, Ulrike
        http://www.kyb.mpg.de/fileadmin/user_upload/files/publications/attachments/Luxburg07_tutorial_4488%5b0%5d.pdf

    """

    """
    If normed=True, L = D^(-1/2) * (D - A) * D^(-1/2) else L = D - A.
    normed=True is recommended.
    """
    #normed_laplacian, dd = graph_laplacian(affinity_matrix, normed=True, return_diag=True)
    normed_laplacian, dd = csgraph.laplacian(affinity_matrix, normed=True, return_diag=True)
    laplacian = _set_diag(normed_laplacian, 1, True)

    """
    n_components size is N - 1.
    Setting N - 1 may lead to slow execution time...
    """
    n_components = affinity_matrix.shape[0] - 1

    """
    shift-invert mode
    The shift-invert mode provides more than just a fast way to obtain a few small eigenvalues.
    http://docs.scipy.org/doc/scipy/reference/tutorial/arpack.html

    The normalized Laplacian has eigenvalues between 0 and 2.
    I - L has eigenvalues between -1 and 1.
    """
    eigenvalues, eigenvectors = eigsh(-laplacian, k=n_components, which="LM", sigma=1.0, maxiter=5000)
    eigenvalues = -eigenvalues[::-1]  # Reverse and sign inversion.

    plt.title('Largest eigen values of input matrix')
    plt.scatter(np.arange(len(eigenvalues)), eigenvalues)
    plt.grid()

    max_gap = 0
    gap_pre_index = 0
    for i in range(1, eigenvalues.size):
        gap = eigenvalues[i] - eigenvalues[i - 1]
        if gap > max_gap:
            max_gap = gap
            gap_pre_index = i - 1

    k = gap_pre_index + 1

    return k


def agglomerative_clustering_on_deltas(summed_delta_matrix: np.ndarray, cluster_count = None, distance_threshold=None, linkage='complete') -> np.ndarray:
  #linkage 'ward' does not work with precomputed dist mat
  normed_distance_matrix = normed_values(summed_delta_matrix)

  if cluster_count is None and distance_threshold is None:
    distance_threshold = 0.1

  if cluster_count is not None:

    clustering = AgglomerativeClustering(n_clusters=cluster_count, affinity='precomputed', linkage=linkage)
  else:
    clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=distance_threshold, affinity='precomputed', linkage=linkage)

  result = clustering.fit(normed_distance_matrix)
  #return result.labels_, cluster_count, normed_distance_matrix
  return result.labels_



def agglomerative_clustering_on_deltas_ward(summed_delta_matrix: np.ndarray, cluster_count: Num, linkage='complete') -> np.ndarray:
  #linkage 'ward' does not work with precomputed dist mat
  normed_distance_matrix = normed_values(summed_delta_matrix)

  if cluster_count is None:
    distance_threshold = 0.1



  if cluster_count is not None:

    clustering = AgglomerativeClustering(n_clusters=cluster_count, affinity='precomputed', linkage=linkage)
  else:
    clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=distance_threshold, affinity='precomputed', linkage=linkage)

  result = clustering.fit(normed_distance_matrix)
  return result.labels_, cluster_count, normed_distance_matrix





def agglomerative_clustering_on_deltas_var(summed_delta_matrix: np.ndarray, cluster_count: Num, linkage='complete') -> np.ndarray:
  #linkage 'ward' not with dist mat
  normed_distance_matrix = normed_values(summed_delta_matrix)

  if cluster_count is None:
    distance_threshold = 0.1



  if cluster_count is not None:

    clustering = AgglomerativeClustering(n_clusters=cluster_count, affinity='precomputed', linkage=linkage)
  else:
    clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=distance_threshold, affinity='precomputed', linkage=linkage)

  result = clustering.fit(normed_distance_matrix)
  return result.labels_, cluster_count, normed_distance_matrix




def agglomerative_clustering_on_deltas(summed_delta_matrix: np.ndarray, cluster_count: Num, linkage='average') -> np.ndarray:
  normed_distance_matrix = normed_values(summed_delta_matrix)
  clustering = AgglomerativeClustering(n_clusters=cluster_count, affinity='precomputed', linkage=linkage)
  result = clustering.fit(normed_distance_matrix)
  return result.labels_

def calculate_average_value(delta_matrix: np.ndarray) -> Num:
  return np.sum(delta_matrix) / (len(delta_matrix[0])*len(delta_matrix[0]))

def calculate_knn_graph(delta_matrix: np.ndarray) -> Num:
  return skn.kneighbors_graph(X=delta_matrix, n_neighbors=10, mode='distance', include_self=True).toarraly()

#@timing
def calculate_distance_matrix(traj_array: np.ndarray) -> np.ndarray:
  '''
  traj_array: Array containing positions x, y, z of all atoms in time step i
  '''
  #return scipy.spatial.distance_matrix(traj_array, traj_array, p=2)
  return np.linalg.norm(traj_array[:, None, :] - traj_array[None, :, :], axis=-1)

#@timing
def calculate_delta_matrix(dist_then: np.ndarray, dist_now: np.ndarray) -> np.ndarray:
  '''
  dist_then: Distance matrix in time step i
  dist_now: Distance matrix in time step i+1
  Returns the delta matrix and the calculatet clusters
  '''
  return np.absolute(np.subtract(dist_now, dist_then))

def sum_up_matrices(matrices: np.ndarray) -> np.ndarray:
    return reduce(np.add, matrices)

def normed_values(arr: np.ndarray):
  return arr / arr.max()


def row_normalization(matrix):

    row_sums = matrix.sum(axis=1, keepdims=True)
    normalized_matrix = matrix / row_sums
    return normalized_matrix

def symmetrize(matrix):

    symmetrized_matrix = (matrix + matrix.T) / 2
    return symmetrized_matrix

def row_normalization_followed_by_symmetrization(matrix):

    normalized_matrix = row_normalization(matrix)

    symmetrized_matrix = symmetrize(normalized_matrix)
    
    return symmetrized_matrix