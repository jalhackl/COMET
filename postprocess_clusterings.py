import numpy as np
from sklearn.neighbors import NearestNeighbors


def assign_noise_points(distance_matrix, cluster_labels, noise_label=-1, n_neighbors=1):
    # Identify noise points (noise_label / -1) and clustered points
    noise_indices = np.where(cluster_labels == noise_label)[0]
    clustered_indices = np.where(cluster_labels != noise_label)[0]

    # Extract clustered labels and corresponding data
    clustered_labels = cluster_labels[clustered_indices]

    if len(clustered_indices) > 0 and len(noise_indices) > 0:
        # Use nearest neighbor search on the clustered points
        nn = NearestNeighbors(n_neighbors=n_neighbors, metric="precomputed")  # Using the distance matrix provided as input
        nn.fit(distance_matrix[np.ix_(clustered_indices, clustered_indices)])

        # Find the nearest clustered neighbor for each noise point
        distances, nearest_clustered_idx = nn.kneighbors(distance_matrix[np.ix_(noise_indices, clustered_indices)])

        # Assign each noise point the label of its nearest clustered neighbor
        cluster_labels[noise_indices] = clustered_labels[nearest_clustered_idx.flatten()]

    return cluster_labels
