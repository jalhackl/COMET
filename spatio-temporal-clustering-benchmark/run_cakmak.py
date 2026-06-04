# control execution time of functions
import threading
import time
import numpy as np
import pandas as pd
import os
import logging
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import (
    silhouette_score,
    adjusted_mutual_info_score,
    normalized_mutual_info_score,
    adjusted_rand_score
)
from st_clustering import ST_DBSCAN, ST_Agglomerative, ST_BIRCH, ST_HDBSCAN

# =======================
# CONFIGURATION
# =======================
TIMER = 200
PERMUT = 12
RUNTIME_CUTOFF = 200.0  # seconds

PATH = 'dataset_files'
PATH = "/COMET/full_data"
FRAME_SIZE = 100
FRAME_OVERLAP = 10

# =======================
# TIMEOUT INFRASTRUCTURE
# =======================
class TimeoutError(Exception):
    pass


class InterruptableThread(threading.Thread):
    def __init__(self, func, *args, **kwargs):
        super().__init__()
        self._func = func
        self._args = args
        self._kwargs = kwargs
        self._result = None

    def run(self):
        self._result = self._func(*self._args, **self._kwargs)

    @property
    def result(self):
        return self._result


class timeout(object):
    def __init__(self, sec):
        self._sec = sec

    def __call__(self, f):
        def wrapped_f(*args, **kwargs):
            it = InterruptableThread(f, *args, **kwargs)
            it.start()
            it.join(self._sec)
            if not it.is_alive():
                return it.result
            raise TimeoutError("execution expired")
        return wrapped_f


# =======================
# UTILITIES
# =======================
def make_generator(parameters):
    if not parameters:
        yield dict()
    else:
        key = list(parameters.keys())[0]
        rest = {p: parameters[p] for p in parameters if p != key}
        for val in parameters[key]:
            for pars in make_generator(rest):
                temp = pars.copy()
                temp[key] = val
                yield temp


def log_or_timeout(filename, method, nmi, ari, runtime):
    if runtime > RUNTIME_CUTOFF:
        logging.info(
            f"dataset: {filename}, method: {method}, TIMEOUT"
        )
        return False
    else:
        logging.info(
            f"dataset: {filename}, method: {method}, "
            f"nmi: {round(nmi,4)}, ari: {round(ari,4)}, runtime: {round(runtime,4)}"
        )
        return True


# =======================
# GRID SEARCH
# =======================
@timeout(TIMER * PERMUT)
def st_grid_search(estimator, split, X, param_dict, feature,
                   y=None, frame_size=None, frame_overlap=None):

    param_opt = None
    s_max = -np.inf

    for param in make_generator(param_dict):
        clust = estimator(**param)

        if split:
            clust.st_fit_frame_split(X, frame_size, frame_overlap)
        else:
            clust.st_fit(X)

        try:
            if feature == "nmi":
                score = normalized_mutual_info_score(y, clust.labels)
            elif feature == "ari":
                score = adjusted_rand_score(y, clust.labels)
            else:
                raise ValueError(feature)
        except Exception:
            continue

        if score > s_max:
            s_max = score
            param_opt = param

    return param_opt


# =======================
# TEST WRAPPER
# =======================
class Test(object):

    @timeout(TIMER)
    def frame_split_cluster(self, algorithm, data, frame_size, frame_overlap, true_labels):
        start = time.time()
        algorithm.st_fit_frame_split(data, frame_size, frame_overlap)
        runtime = time.time() - start
        ari = adjusted_rand_score(true_labels, algorithm.labels)
        nmi = normalized_mutual_info_score(true_labels, algorithm.labels)
        return nmi, ari, runtime


# =======================
# LOGGING
# =======================
# =======================
# LOGGING
# =======================
LOG_FILE = "cakmak_results_complete.csv"
LOG_FILE = "cakmak_results_birch.csv"

logging.basicConfig(
    level=logging.INFO,
    filename=LOG_FILE,
    filemode="a",
    format="%(asctime)s - %(message)s"
)

# Read already processed datasets
processed_datasets = set()
if os.path.exists(LOG_FILE):
    with open(LOG_FILE, "r") as f:
        for line in f:
            # Assumes log lines start with "dataset: <filename>, ..."
            if "dataset:" in line:
                # Extract the filename
                parts = line.split("dataset:")[1].split(",")[0].strip()
                processed_datasets.add(parts)

# =======================
# MAIN LOOP
# =======================
dataset_files = os.listdir(PATH)
t = Test()

not_timed_out_dbscan = True
not_timed_out_agglo = True
not_timed_out_birch = True
not_timed_out_hdbscan = True

for ds in dataset_files:
    filename = ds

    if filename in processed_datasets:
        print(f"Skipping already processed dataset: {filename}")
        continue

    print(f"\nProcessing dataset: {filename}")

    df = pd.read_csv(os.path.join(PATH, filename))
    df["x"] = (df["x"] - df["x"].min()) / (df["x"].max() - df["x"].min())
    df["y"] = (df["y"] - df["y"].min()) / (df["y"].max() - df["y"].min())

    data = df[["frame", "x", "y"]].values
    labels = df["cid"].to_numpy()

    n_cluster = len(np.unique(labels)) - (1 if -1 in labels else 0)

    param_dict_dbscan = {
        "eps1": [0.03, 0.05],
        "eps2": [25, 50],
        "min_samples": [2]
        # "eps1": [0.02, 0.03, 0.04, 0.05],
        # "eps2": [5, 25, 50, 100],
    }

    param_dict_hdbscan = {
    "eps2": [25, 50],
    "min_cluster_size": [n_cluster],
    "min_samples": [2]}

    param_dict_agglo = {
        "eps2": [50, 100],
        "n_clusters": [n_cluster]
    }

    param_dict_birch = {
        "eps2": [50, 100],
        "threshold": [0.5],
        "n_clusters": [n_cluster]
    }

    # =======================
    # ST_DBSCAN
    # =======================
    if not_timed_out_dbscan:
        try:
            opt = st_grid_search(
                ST_DBSCAN, True, data, param_dict_dbscan,
                feature="nmi", y=labels,
                frame_size=FRAME_SIZE, frame_overlap=FRAME_OVERLAP
            )
            algo = ST_DBSCAN(**opt)
            nmi, ari, runtime = t.frame_split_cluster(
                algo, data, FRAME_SIZE, FRAME_OVERLAP, labels
            )
            log_or_timeout(filename, "DBSCAN", nmi, ari, runtime)

        except TimeoutError:
            logging.info(f"dataset: {filename}, method: DBSCAN, TIMEOUT")
            not_timed_out_dbscan = False

        except Exception:
            logging.info(f"dataset: {filename}, method: DBSCAN, ERROR")

    # =======================
    # ST_Agglomerative
    # =======================
    if not_timed_out_agglo:
        try:
            opt = st_grid_search(
                ST_Agglomerative, True, data, param_dict_agglo,
                feature="nmi", y=labels,
                frame_size=FRAME_SIZE, frame_overlap=FRAME_OVERLAP
            )
            algo = ST_Agglomerative(**opt)
            nmi, ari, runtime = t.frame_split_cluster(
                algo, data, FRAME_SIZE, FRAME_OVERLAP, labels
            )
            log_or_timeout(filename, "Agglomerative", nmi, ari, runtime)

        except TimeoutError:
            logging.info(f"dataset: {filename}, method: Agglomerative, TIMEOUT")
            not_timed_out_agglo = False

        except Exception:
            logging.info(f"dataset: {filename}, method: Agglomerative, ERROR")

    # =======================
    # ST_BIRCH
    # =======================
    if not_timed_out_birch:
        try:
            opt = st_grid_search(
                ST_BIRCH, True, data, param_dict_birch,
                feature="nmi", y=labels,
                frame_size=FRAME_SIZE, frame_overlap=FRAME_OVERLAP
            )
            algo = ST_BIRCH(**opt)
            nmi, ari, runtime = t.frame_split_cluster(
                algo, data, FRAME_SIZE, FRAME_OVERLAP, labels
            )
            log_or_timeout(filename, "BIRCH", nmi, ari, runtime)

        except TimeoutError:
            logging.info(f"dataset: {filename}, method: BIRCH, TIMEOUT")
            not_timed_out_birch = False

        except Exception:
            logging.info(f"dataset: {filename}, method: BIRCH, ERROR")


# =======================
# ST_HDBSCAN
# =======================
    if not_timed_out_hdbscan:
        try:
            opt = st_grid_search(
                ST_HDBSCAN,
                True,
                data,
                param_dict_hdbscan,
                feature="nmi",
                y=labels,
                frame_size=FRAME_SIZE,
                frame_overlap=FRAME_OVERLAP
            )

            algo = ST_HDBSCAN(**opt)

            nmi, ari, runtime = t.frame_split_cluster(
                algo,
                data,
                FRAME_SIZE,
                FRAME_OVERLAP,
                labels
            )

            log_or_timeout(filename, "HDBSCAN", nmi, ari, runtime)

        except TimeoutError:
            logging.info(f"dataset: {filename}, method: HDBSCAN, TIMEOUT")
            not_timed_out_hdbscan = False

        except Exception:
            logging.info(f"dataset: {filename}, method: HDBSCAN, ERROR")


        print(f"Finished processing dataset: {filename}")
