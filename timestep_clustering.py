import seaborn as sns
import umap
import numpy as np
from scipy.stats import wasserstein_distance
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import ruptures as rpt


def plot_distance_heatmap(distance_matrix, title="Timestep heatmap"):
    plt.figure(figsize=(10, 8))
    sns.heatmap(distance_matrix, cmap='viridis', annot=False, cbar=True)
    plt.title(title, fontsize=16)
    plt.xlabel("Matrix Index", fontsize=14)
    plt.ylabel("Matrix Index", fontsize=14)
    plt.show()

# Define the Wasserstein distance as a custom metric
def wasserstein_distance_matrices(D1, D2):
    """
    Compute the Wasserstein distance between two distance matrices.
    This metric is permutation-invariant.
    """
    # Flatten and sort the matrices
    v1 = np.sort(D1.flatten())
    v2 = np.sort(D2.flatten())
    # Compute the Wasserstein distance
    return wasserstein_distance(v1, v2)


# Define a custom distance function for two distance matrices
def frobenius_distance(D1, D2):
    """
    Compute the Frobenius distance between two distance matrices.
    D1 and D2 are assumed to be 2D numpy arrays representing distance matrices.
    """
    return np.sqrt(np.sum((D1 - D2)**2))



def compute_timstep_clustering(input_matrices, title_type="Delta-matrices", dim_red="umap", metric="wasserstein", plot_dim_red=True, kde_plot=True, plot_heatmap=True):



    distance_matrices = input_matrices
    n_matrices = len(distance_matrices)
    from joblib import Parallel, delayed

    # Compute pairwise distances in parallel
    def compute_distance_frobenius(i, j):
        return frobenius_distance(distance_matrices[i], distance_matrices[j])
    
    # Compute pairwise distances in parallel
    def compute_distance_wasserstein(i, j):
        return wasserstein_distance_matrices(distance_matrices[i], distance_matrices[j])
    
    pairwise_distances = np.zeros((n_matrices, n_matrices))

    if metric == "wasserstein":
        results = Parallel(n_jobs=-1)(
            delayed(compute_distance_frobenius)(i, j) for i in range(n_matrices) for j in range(i, n_matrices)
        )
    elif metric == "frobenius":
        results = Parallel(n_jobs=-1)(
            delayed(compute_distance_wasserstein)(i, j) for i in range(n_matrices) for j in range(i, n_matrices)
        )
    else:
        raise "Metric unknown!"
    


    # Fill the symmetric distance matrix
    for idx, (i, j) in enumerate([(i, j) for i in range(n_matrices) for j in range(i, n_matrices)]):
        pairwise_distances[i, j] = results[idx]
        pairwise_distances[j, i] = results[idx]

    dim_red_valid = True
    if dim_red == "umap":
        # Initialize and fit UMAP with the precomputed pairwise distance matrix
        reducer = umap.UMAP(metric="precomputed", random_state=42)
        embedding = reducer.fit_transform(pairwise_distances)
        colors = np.arange(len(embedding))
        
    elif dim_red == "tsne":
        # Apply t-SNE with the precomputed distance matrix
        tsne = TSNE(metric="precomputed", perplexity=20, init="random")
        embedding = tsne.fit_transform(pairwise_distances)
        colors = np.arange(len(embedding))

    else:
        print("Dimensionality reduction method unknown or not set!")
        dim_red_valid = False

    # Plot the resulting 2D embedding
    if plot_dim_red and dim_red_valid:
        # Plot the results
        plt.scatter(embedding[:, 0], embedding[:, 1], c=colors, cmap='viridis')
        plt.title(dim_red + " with " + metric)
        plt.xlabel(dim_red + " Dimension 1")
        plt.ylabel(dim_red + " Dimension 2")
        plt.show()
    
    
    if kde_plot:
        # Create a density plot with kdeplot
        plt.figure(figsize=(8, 6))
        ax = plt.gca()

        # Create the KDE density plot with contours
        sns.kdeplot(x=embedding[:, 0], y=embedding[:, 1], cmap='inferno', fill=True, levels=20, alpha=0.7, ax=ax)

        # Add the colorbar
        sm = plt.cm.ScalarMappable(cmap='inferno', norm=plt.Normalize(vmin=0, vmax=2e-4))
        sm.set_array([])

        # Create a colorbar based on the scalar mappable object
        plt.colorbar(sm, label='PDF', ax=ax)

        # Label axes
        plt.xlabel('$z_1$', fontsize=14)
        plt.ylabel('$z_2$', fontsize=14)

        plt.show()

    if plot_heatmap:
        plot_distance_heatmap(pairwise_distances, title_type + "\n Timestep Heatmap, " + dim_red + ", " + metric)

    return embedding, pairwise_distances

# one can apply clustering to the embeddings (or, in principle, to the pairwise_distances-matrix)
# There are various reasonable alternatives: apply a time-series change detection algorithm, apply tda etc.
# for tda, see tda_fkt.py

def apply_rpt_change_detection(input_values, kernel="rbf", pen=1):
    algo = rpt.KernelCPD(kernel=kernel).fit(input_values)
    change_points = algo.predict(pen=pen)
    return change_points





def compute_lcss_weights(dists, epsilon, return_additional_statistics=False):


    start_index = 0
    weights = np.zeros(len(dists))
    segments = np.zeros(len(dists))
    weights_refined = np.zeros(len(dists))   

    index = 1  # Start at 1 to prevent empty slice in np.var()
    segment = 0
    
    while index < len(dists):
        if np.var(dists[start_index:index+1]) <= epsilon:
            index += 1
        else:
            # Assign segment length as weight
            weights[start_index:index] = index - start_index

            # Assign segment IDs
            segments[start_index:index] = segment

            # Compute refined weight (e.g., segment variance)
            weights_refined[start_index:index] = np.var(dists[start_index:index])

            # Move to the next segment
            segment += 1
            start_index = index
            index += 1  # Move forward

    # Assign the last segment
    weights[start_index:index] = index - start_index
    segments[start_index:index] = segment
    weights_refined[start_index:index] = np.var(dists[start_index:index])

    if return_additional_statistics:
        return weights, segments, weights_refined
    else:
        return weights, segments