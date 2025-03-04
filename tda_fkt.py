
import numpy as np
from ripser import ripser
from sklearn.metrics.pairwise import rbf_kernel, euclidean_distances


def compute_persistent_homology_knn_distance_matrix(x, k, use_median=False, exclude_self=False):
    """
    Description:
        computation of persistent homology (using scikit-TDA/ripser)
    Arguments:
        x numpy.ndarray: data points
    Returns:
        D
        cocycles
        diagrams
    """

    newdist = knn_distance_matrix(x, k, use_median=use_median, exclude_self=exclude_self)
    result = ripser(newdist, do_cocycles=True, distance_matrix=True)

        
    #result = ripser(x, do_cocycles=True)
    diagrams = result['dgms']
    cocycles = result['cocycles']
    D = result['dperm2all']
    
    return D, cocycles, diagrams


def compute_persistent_homology_rbf_kernel(x, sigma=1):
    """
    Description:
        computation of persistent homology (using scikit-TDA/ripser)
    Arguments:
        x numpy.ndarray: data points
    Returns:
        D
        cocycles
        diagrams
    """
    from sklearn.metrics.pairwise import rbf_kernel

    newdist = rbf_kernel(x, gamma=1.0 / (2 * sigma ** 2))
    newdist = np.sqrt(2 * (1 - newdist))


    result = ripser(newdist, do_cocycles=True, distance_matrix=True)

        
    #result = ripser(x, do_cocycles=True)
    diagrams = result['dgms']
    cocycles = result['cocycles']
    D = result['dperm2all']
    
    return D, cocycles, diagrams





def compute_persistent_homology(x):
    """
    Description:
        computation of persistent homology (using scikit-TDA/ripser)
    Arguments:
        x numpy.ndarray: data points
    Returns:
        D
        cocycles
        diagrams
    """
        
    result = ripser(x, do_cocycles=True)
    diagrams = result['dgms']
    cocycles = result['cocycles']
    D = result['dperm2all']
    
    return D, cocycles, diagrams
	
	
	
def compute_persistent_homology(x):
    """
    Description:
        computation of persistent homology (using scikit-TDA/ripser)
    Arguments:
        x numpy.ndarray: data points
    Returns:
        D
        cocycles
        diagrams
    """
        
    result = ripser(x, do_cocycles=True)
    diagrams = result['dgms']
    cocycles = result['cocycles']
    D = result['dperm2all']
    
    return D, cocycles, diagrams



def make_symmetric_matrix(matrix):
    lower_tri = np.tril(matrix, k=-1)  # Extract lower triangular part
    upper_tri = np.triu(matrix, k=1)   # Extract upper triangular part
    symmetric_matrix = (lower_tri + upper_tri) / 2  # Calculate average
    symmetric_matrix = symmetric_matrix + symmetric_matrix.T  # Reflect upper triangular part to make it symmetric
    return symmetric_matrix

def knn_distance_matrix(data, k, make_symmetric = True, use_median=True, exclude_self=False):
    from sklearn.metrics.pairwise import euclidean_distances
    # Compute pairwise Euclidean distances
    pairwise_distances = euclidean_distances(data)

    # Sort distances to find k-nearest neighbors
    sorted_distances_indices = np.argsort(pairwise_distances, axis=1)[:, :k+1]


    distance_matrix = np.zeros((len(data), len(data)))
    for i in range(len(data)):
        for j in range(len(data)):
            if i == j:
                distance_matrix[i][j] = 0
            else:
                next_items = sorted_distances_indices[j]

                further_distances = pairwise_distances[i][next_items]
                if exclude_self:
                    further_distances = further_distances[1:]

                if not use_median:
                    mean_distance = np.mean(further_distances)
                else:
                    mean_distance = np.median(further_distances)

                distance_matrix[i][j] = mean_distance

    if make_symmetric:
        distance_matrix = make_symmetric_matrix(distance_matrix)

    return distance_matrix




