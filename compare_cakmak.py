import numpy as np
import pandas as pd
import os
import time
import logging
import json
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import silhouette_score, adjusted_mutual_info_score
from st_clustering_benchmark_modified import ST_DBSCAN, ST_Agglomerative, ST_KMeans, ST_OPTICS, ST_SpectralClustering, ST_AffinityPropagation, ST_BIRCH, ST_HDBSCAN


def format_cluster_df(df, substitutions = {'t':'frame', 'obj_id':'id','label':'cid','x':'x','y':'y'}, add_z=True):
    filtered_df = df[list(substitutions.values())]

    filtered_df = filtered_df.rename(columns={v: k for k, v in substitutions.items()})

    if add_z:
        if 'z' not in df.columns:
            filtered_df['z'] = 0


    return filtered_df


# Code from Cakmak 2021 repository

# control execution time of functions
import threading

TIMER = 120
PERMUT = 12

class TimeoutError(Exception):
    pass

class InterruptableThread(threading.Thread):
    def __init__(self, func, *args, **kwargs):
        threading.Thread.__init__(self)
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
            raise TimeoutError('execution expired')
        return wrapped_f
    


def make_generator(parameters):
    """Helper function for st_grid_search. Returns a dictionary of all possible parameter combinations."""
    if not parameters:
        yield dict()
    else:
        key_to_iterate = list(parameters.keys())[0]
        next_round_parameters = {p : parameters[p]
                    for p in parameters if p != key_to_iterate}
        for val in parameters[key_to_iterate]:
            for pars in make_generator(next_round_parameters):
                temp_res = pars
                temp_res[key_to_iterate] = val
                yield temp_res
                
def st_silhouette_score(X, labels, eps1=0.05, eps2=10, metric='euclidean'):
    """Helper function for st_grid_search. Hyperparameter combinations are evaluated with the Silhouette score."""
    n, m = X.shape
    time_dist = pdist(X[:, 0].reshape(n, 1), metric=metric)
    euc_dist = pdist(X[:, 1:], metric=metric)

    # filter the euc_dist matrix using the time_dist
    dist = np.where(time_dist <= eps2, euc_dist, 2 * eps1)

    return silhouette_score(squareform(dist), labels, metric='precomputed')

@timeout(TIMER*PERMUT)
def st_grid_search(estimator, split, X, param_dict, metric, y=None, frame_size=None, frame_overlap=None):
    """
    Grid Search of hyperparameters for spatial-temporal clustering algorithms
    
    Parameters
    ----------
    estimator: class
        ST clustering algorithm
    split: boolean
        Flag to indicate whether whole X should be loaded in RAM or processed in smaller chunks.
    X: numpy array
        Data on which grid search is performed
    param_dict: dict
        Dictionary with parameters to be optimized as keys and value range of grid search as value.
    metric: str
        The metric to evaluate the clustering quality
    y: numpy array
        Optional. Some metrics compare predictions with ground truth. Then, labels need to be provided.
    frame_size: int
        Optional. If split is True, indicate how large the chunks should be.
    
    Returns
    -------
    param_opt
        Optimal hyperparameter combination
    """
    param_opt = None
    s_max = 0
    for param in make_generator(param_dict):
        clust = estimator(**param)
        if not split:
            clust.st_fit(X)
        else:
            clust.st_fit_frame_split(X, frame_size, frame_overlap)
            
        if param_opt is None: 
            param_opt = param
        
        # different performance evaluation metrics
        if metric=='silhouette':
            try:
                score = st_silhouette_score(X=X, labels=clust.labels, eps1=param['eps1'] , eps2=param['eps2'], metric='euclidean')
            except (TypeError, ValueError) as e:
                continue
            #print('Silhouette score for parameters {}: {}'.format(param,score))
        elif metric=='ami':
            score = adjusted_mutual_info_score(y,clust.labels)

        # store parameter combination if it outperforms given the metric
        if score > s_max:
            s_max = score
            param_opt = param
    return param_opt

@timeout(TIMER*PERMUT)
def traj_grid_search(estimator, X, param_dict, metric):
    """
    Grid Search of hyperparameters for spatial-temporal clustering algorithms
    
    Parameters
    ----------
    estimator: class
        ST clustering algorithm
    split: boolean
        Flag to indicate whether whole X should be loaded in RAM or processed in smaller chunks.
    X: numpy array
        Data on which grid search is performed
    param_dict: dict
        Dictionary with parameters to be optimized as keys and value range of grid search as value.
    metric: str
        The metric to evaluate the clustering quality
    y: numpy array
        Optional. Some metrics compare predictions with ground truth. Then, labels need to be provided.
    frame_size: int
        Optional. If split is True, indicate how large the chunks should be.
    
    Returns
    -------
    param_opt
        Optimal hyperparameter combination
    """
    param_opt = {'detect_radius':40, 'similarity_threshold':0.5}
    s_max = 0
    for param in make_generator(param_dict):
        clust = estimator(**param)
        clust.st_fit(X)
        
        if param_opt is None: 
            param_opt = param
        
        # different performance evaluation metrics
        if metric=='silhouette':
            try:
                score = st_silhouette_score(X=X, labels=clust.labels, eps1=param['eps1'] , eps2=param['eps2'], metric='euclidean')
            except (TypeError, ValueError) as e:
                continue
            #print('Silhouette score for parameters {}: {}'.format(param,score))
        elif metric=='ami':
            score = adjusted_mutual_info_score(clust.true_labels,clust.labels)
            #print('AMI score for parameters {}: {}'.format(param,score))
            
        # store parameter combination if it outperforms given the metric
        if score > s_max:
            s_max = score
            param_opt = param
    return param_opt
