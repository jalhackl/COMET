import numpy as np
from scipy.spatial.distance import pdist, squareform

from sklearn.cluster import (
    DBSCAN,
    AgglomerativeClustering,
    KMeans,
    Birch
)
from sklearn.utils import check_array

import hdbscan


# ======================================================
# ST DECORATOR
# ======================================================
def st_decorator(target):

    def st_fit(self, X):
        """
        Apply the ST clustering algorithm
        """
        X = check_array(X)

        if self.eps1 <= 0.0 or self.eps2 <= 0.0:
            raise ValueError("eps1, eps2 must be positive")

        n, _ = X.shape

        # time distance
        time_dist = pdist(X[:, 0].reshape(n, 1), metric=self.dist)

        # spatial distance
        euc_dist = pdist(X[:, 1:], metric=self.dist)

        # spatio-temporal distance
        dist = np.where(time_dist <= self.eps2, euc_dist, 2 * self.eps1)

        # uniform call for all algorithms (INCLUDING HDBSCAN)
        self.fit(squareform(dist))

        self.labels = self.labels_
        return self

    def st_fit_frame_split(self, X, frame_size, frame_overlap=None):
        """
        Apply ST clustering with frame splitting
        """
        X = check_array(X)

        if frame_overlap is None:
            frame_overlap = self.eps2

        if self.eps1 <= 0.0 or self.eps2 <= 0.0:
            raise ValueError("eps1, eps2 must be positive")

        if frame_size <= 0.0 or frame_overlap <= 0.0 or frame_size < frame_overlap:
            raise ValueError("frame_size, frame_overlap not correctly configured")

        time = np.unique(X[:, 0])

        labels = None
        right_overlap = 0

        for i in range(0, len(time), (frame_size - frame_overlap + 1)):
            period = time[i:i + frame_size]
            frame = X[np.isin(X[:, 0], period)]

            self.st_fit(frame)

            if labels is None:
                labels = self.labels
            else:
                frame_one_overlap_labels = labels[len(labels) - right_overlap:]
                frame_two_overlap_labels = self.labels[:right_overlap]

                mapper = dict(zip(frame_two_overlap_labels, frame_one_overlap_labels))

                ignore_clusters = set(self.labels) - set(frame_two_overlap_labels)

                new_labels = [
                    mapper[l] if l not in ignore_clusters else -99
                    for l in self.labels
                ]

                labels = labels[:-right_overlap]
                labels = np.concatenate((labels, new_labels))

            right_overlap = len(
                X[np.isin(X[:, 0], period[-frame_overlap + 1:])]
            )

            if i + frame_size > max(time):
                break

        self.labels = labels
        return self

    target.st_fit = st_fit
    target.st_fit_frame_split = st_fit_frame_split
    return target


# ======================================================
# ST_DBSCAN
# ======================================================
@st_decorator
class ST_DBSCAN(DBSCAN):
    def __init__(
        self,
        eps1=0.5,
        eps2=10,
        min_samples=5,
        metric="precomputed",
        n_jobs=-1,
        algorithm="auto",
        leaf_size=30,
        metric_params=None,
        p=None,
        dist="euclidean"
    ):
        self.eps = eps1
        self.eps1 = eps1
        self.eps2 = eps2
        self.min_samples = min_samples
        self.metric = metric
        self.n_jobs = n_jobs
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.metric_params = metric_params
        self.p = p
        self.dist = dist


# ======================================================
# ST_AGGLOMERATIVE
# ======================================================
@st_decorator
class ST_Agglomerative(AgglomerativeClustering):
    def __init__(
        self,
        eps1=0.5,
        eps2=10,
        n_clusters=2,
        *,
        affinity="precomputed",
        memory=None,
        connectivity=None,
        compute_full_tree="auto",
        linkage="average",
        distance_threshold=None,
        compute_distances=False,
        dist="euclidean",
        metric="precomputed"
    ):
        self.eps1 = eps1
        self.eps2 = eps2
        self.n_clusters = n_clusters
        self.affinity = affinity
        self.memory = memory
        self.connectivity = connectivity
        self.compute_full_tree = compute_full_tree
        self.linkage = linkage
        self.distance_threshold = distance_threshold
        self.compute_distances = compute_distances
        self.metric = metric
        self.dist = dist


# ======================================================
# ST_KMEANS
# ======================================================
@st_decorator
class ST_KMeans(KMeans):
    def __init__(
        self,
        eps1=0.5,
        eps2=10,
        n_clusters=8,
        *,
        init="k-means++",
        n_init=10,
        max_iter=300,
        tol=1e-4,
        verbose=0,
        random_state=None,
        copy_x=True,
        algorithm="auto",
        dist="euclidean"
    ):
        self.eps1 = eps1
        self.eps2 = eps2
        self.n_clusters = n_clusters
        self.init = init
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.random_state = random_state
        self.copy_x = copy_x
        self.algorithm = algorithm
        self.dist = dist


# ======================================================
# ST_BIRCH
# ======================================================
@st_decorator
class ST_BIRCH(Birch):
    def __init__(
        self,
        eps1=0.5,
        eps2=10,
        threshold=0.5,
        branching_factor=50,
        n_clusters=3,
        compute_labels=True,
        copy=True,
        dist="euclidean"
    ):
        self.eps1 = eps1
        self.eps2 = eps2
        self.threshold = threshold
        self.branching_factor = branching_factor
        self.n_clusters = n_clusters
        self.compute_labels = compute_labels
        self.copy = copy
        self.dist = dist


# ======================================================
# ST_HDBSCAN (FIXED)
# ======================================================
@st_decorator
class ST_HDBSCAN(hdbscan.HDBSCAN):
    def __init__(
        self,
        eps1=0.5,
        eps2=10,
        dist="euclidean",
        metric="precomputed",
        gen_min_span_tree=False,
        **kwargs
    ):
        # ST-specific
        self.eps1 = eps1
        self.eps2 = eps2
        self.dist = dist

        # Proper HDBSCAN initialization
        super().__init__(
            metric=metric,
            gen_min_span_tree=gen_min_span_tree,
            **kwargs
        )
