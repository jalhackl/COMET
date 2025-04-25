import seaborn as sns
import umap
import numpy as np
from scipy.stats import wasserstein_distance
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import ruptures as rpt


def plot_distance_heatmap(distance_matrix, title="Timestep heatmap"):
    plt.figure(figsize=(10, 8))
    sns.heatmap(distance_matrix, cmap="viridis", annot=False, cbar=True)
    plt.title(title, fontsize=16)
    plt.xlabel("Matrix Index", fontsize=14)
    plt.ylabel("Matrix Index", fontsize=14)
    plt.show()


# Define the Wasserstein distance as a custom metric
def wasserstein_distance_matrices(D1, D2, sort=True):
    """
    Compute the Wasserstein distance between two distance matrices.
    This metric is permutation-invariant.
    """
    # Flatten and sort the matrices
    if sort:
        v1 = np.sort(D1.flatten())
        v2 = np.sort(D2.flatten())
    else:
        v1 = D1.flatten()
        v2 = D2.flatten()
    # Compute the Wasserstein distance
    return wasserstein_distance(v1, v2)


# Define a custom distance function for two distance matrices
def frobenius_distance(D1, D2):
    """
    Compute the Frobenius distance between two distance matrices.
    D1 and D2 are assumed to be 2D numpy arrays representing distance matrices.
    """
    return np.sqrt(np.sum((D1 - D2) ** 2))


def compute_timestep_clustering(
    input_matrices,
    title_type="Delta-matrices",
    dim_red="umap",
    metric="frobenius",
    plot_dim_red=True,
    kde_plot=True,
    plot_heatmap=True,
):

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

    if metric == "frobenius":
        results = Parallel(n_jobs=-1)(
            delayed(compute_distance_frobenius)(i, j)
            for i in range(n_matrices)
            for j in range(i, n_matrices)
        )
    elif metric == "wasserstein":
        results = Parallel(n_jobs=-1)(
            delayed(compute_distance_wasserstein)(i, j)
            for i in range(n_matrices)
            for j in range(i, n_matrices)
        )
    else:
        raise "Metric unknown!"

    # Fill the symmetric distance matrix
    for idx, (i, j) in enumerate(
        [(i, j) for i in range(n_matrices) for j in range(i, n_matrices)]
    ):
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
        plt.scatter(embedding[:, 0], embedding[:, 1], c=colors, cmap="viridis")
        plt.title(dim_red + " with " + metric + " metric")
        plt.xlabel(dim_red + " Dimension 1")
        plt.ylabel(dim_red + " Dimension 2")
        plt.show()

    if kde_plot:
        # Create a density plot with kdeplot
        plt.figure(figsize=(8, 6))
        ax = plt.gca()

        # Create the KDE density plot with contours
        sns.kdeplot(
            x=embedding[:, 0],
            y=embedding[:, 1],
            cmap="inferno",
            fill=True,
            levels=20,
            alpha=0.7,
            ax=ax,
        )

        # Add the colorbar
        sm = plt.cm.ScalarMappable(
            cmap="inferno", norm=plt.Normalize(vmin=0, vmax=2e-4)
        )
        sm.set_array([])

        # Create a colorbar based on the scalar mappable object
        plt.colorbar(sm, label="PDF", ax=ax)

        # Label axes
        plt.xlabel("$z_1$", fontsize=14)
        plt.ylabel("$z_2$", fontsize=14)

        plt.show()

    if plot_heatmap:
        plot_distance_heatmap(
            pairwise_distances,
            title_type + "\n Timestep Heatmap, " + dim_red + ", " + metric,
        )

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
        if np.var(dists[start_index : index + 1]) <= epsilon:
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


def iterative_clustering_approach(
    traj_array,
    delta_matrices,
    Q_values,
    pen=100,
    clustering_method="spectral",
    clustering_params={},
    k_cluster=None,
    compute_Q=False,
):
    import clustering_functions
    from compare_clusterings import max_overlap_matching
    from compare_clusterings import get_RMSD_to_reference

    if k_cluster is not None:
        clustering_params["cluster_count"] = k_cluster

    Q_values = Q_values.T

    change_points = apply_rpt_change_detection(Q_values, pen=pen)

    timestep_clusters = clustering_functions.cluster_timesteps_change_points(
        delta_matrices, change_points, clustering_method, clustering_params
    )
    timestep_clusters = max_overlap_matching(timestep_clusters)

    if compute_Q:
        final_Qs = []
        for timestep in timestep_clusters:
            print(
                f'clustering for timestep {timestep["start"]} - {timestep["end"]}: {timestep["clustering"]}'
            )
            if len(traj_array) > timestep["end"]:
                Q_from_pos_trajectory_average_procrustes_window = get_RMSD_to_reference(
                    traj_array,
                    timestep["clustering"],
                    apply_superimposition="procrustes",
                    return_raw=True,
                    start=timestep["start"],
                    end=timestep["end"],
                )
                final_Qs.append(Q_from_pos_trajectory_average_procrustes_window)

        # Initialize a list to store the concatenated arrays
        concatenated_arrays = []

        # Iterate through the rows of the arrays and concatenate corresponding rows
        if len(final_Qs) > 0:
            for i in range(
                final_Qs[0].shape[0]
            ):  # Assuming all arrays have the same shape
                # Extract the i-th row from each array and concatenate them horizontally (axis=1)
                concatenated_row = np.concatenate([arr[i] for arr in final_Qs], axis=0)
                concatenated_arrays.append(concatenated_row)

            # Convert the list to a numpy array if needed
            concatenated_arrays = np.array(concatenated_arrays)

            plt.figure(figsize=(12, 6))
            for i, cluster_q in enumerate(concatenated_arrays):
                plt.plot(cluster_q, label=f"Cluster {i+1}")

            plt.xlabel("Frame")
            plt.ylabel("RMSD")

            plt.title(
                "RMSD of Clusters for all time segments \n compared to average structure \n Procrustes superimposition"
            )

            plt.legend()

            plt.show()
            return timestep_clusters, final_Qs
        else:
            print("No change points found!")
            return timestep_clusters, []

    return timestep_clusters


def compute_adjacent_distances(matrix, distance_type="wasserstein", aggregation="mean"):
    """
    Compute Wasserstein or Frobenius distances for adjacent rows and aggregate using max or mean.

    Parameters:
        matrix (numpy.ndarray): A 2D NumPy array.
        distance_type (str): "wasserstein" or "frobenius".
        aggregation (str): "max" or "mean".

    Returns:
        np.ndarray: An array of aggregated distances with the same length as input matrix.
    """

    # Compute pairwise distances
    wasserstein_distances = [
        wasserstein_distance(matrix[i], matrix[i + 1]) for i in range(len(matrix) - 1)
    ]
    frobenius_distances = [
        np.linalg.norm(matrix[i] - matrix[i + 1], ord=2) for i in range(len(matrix) - 1)
    ]

    # Select the correct distance type
    distances = (
        wasserstein_distances if distance_type == "wasserstein" else frobenius_distances
    )

    # Aggregated distances for each row
    aggregated_distances = np.zeros(len(matrix))

    for i in range(len(matrix)):
        # Get adjacent distances
        neighbors = []
        if i > 0:
            neighbors.append(distances[i - 1])  # Distance to previous row
        if i < len(matrix) - 1:
            neighbors.append(distances[i])  # Distance to next row

        # Aggregate using max or mean
        if aggregation == "max":
            aggregated_distances[i] = max(neighbors)
        else:  # Default to mean
            aggregated_distances[i] = np.mean(neighbors)

    return aggregated_distances


def distance_outliers(distances, use_otsu=False, upper=90):
    if use_otsu:
        # use otsu's rule to determine threshold
        from skimage.filters import threshold_otsu

        thresh = threshold_otsu(distances)
        outlier_indices = np.where(distances > thresh)[0]
    else:
        Q1, Q3 = np.percentile(distances, [25, upper])
        IQR = Q3 - Q1
        upper_bound = Q3 + 1.5 * IQR
        outlier_indices = np.where(distances > upper_bound)[0]

    outlier_indices_merged = []
    for ie, entry in enumerate(outlier_indices):
        try:
            if outlier_indices[ie + 1] == outlier_indices[ie] + 1:
                continue
            else:
                outlier_indices_merged.append(entry)
        except:
            outlier_indices_merged.append(entry)

    return outlier_indices_merged


def filter_time_windows(change_points, min_length=10):
    filtered = []
    prev = 0

    for cp in change_points:
        if cp - prev >= min_length:
            filtered.append(cp)
        prev = cp

    return filtered
