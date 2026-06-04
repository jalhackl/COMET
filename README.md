# COMET - Clustering of Molecular and Environmental Trajectories

The algorithm allows clustering of spatio-temporal particle data (e.g. molecular dynamics trajectories) using temporal delta matrices and **SHiP** (Hierarchical Clustering with Ultrametric Trees). This repository provides code for preprocessing of general datasets (csv) as well as MD data sets, the COMET clustring framework and benchmarking / visualization functions for analyzing the results and comparing it to several other methods (see main script description below).


COMET computes temporal delta matrices from trajectory data and applies SHiP clustering on these matrices.
Example workflow:

```python
from clustering_functions import clustering_workflow
from SHiP.partitioning import PartitioningMethod as PMethod

matrices_to_apply = ["delta"]
clusterings_to_apply = [{
    "name": "SHiP",
    "method": "ship",
    "params": {
        "partitioning_method": PMethod.ThresholdElbow,
        "hierarchie": 2,
        "tiebreaker_method": "euclidean_distance"
    }
}]

results = clustering_workflow(traj_array, matrices_to_apply, clusterings_to_apply, post_process_noise=True)
```

SHiP (Scalable Hierarchical clustering with Partitioning) is the primary clustering algorithm. It builds ultrametric trees from distance matrices and partitions them using configurable methods (Threshold, Elbow, ThresholdElbow).

```python
from SHiP import SHiP
from SHiP.ultrametric_tree import UltrametricTreeType as UTreeType
from SHiP.partitioning import PartitioningMethod as PMethod

ship = SHiP(data=delta_matrix, treeType=UTreeType.DCTree, is_distance_matrix=True)
labels = ship.fit_predict(hierarchie=2, partitioning_method=PMethod.ThresholdElbow)
```


## Main Scripts

### `cakmak_benchmarks.py`

Runs SHiP+COMET and ST-HDBSCAN on the Cakmak benchmark datasets (Couzin flocks, Reynolds fish, Calovi ants). Processes CSV files from the benchmark, applies mean preprocessing, and evaluates clustering quality with AMI scores. Resumes from existing results via CSV output.

### `cakmak_comparison_small_datasets.ipynb`

Comprehensive comparison notebook for small Cakak benchmark datasets. Runs COMET and ST competitors (ST-HDBSCAN, ST-Spectral, ST-DBSCAN, etc.) with grid search hyperparameter optimization. Includes visualization utilities for delta matrix distributions, silhouette plots, and performance aggregation by dataset size.

### `run_competitors.py`

Runs Cakmak ST clustering competitors (ST-DBSCAN, ST-Agglomerative, ST-KMeans, ST-OPTICS, ST-Spectral, ST-AffinityPropagation, ST-BIRCH, ST-HDBSCAN) on benchmark datasets with timeout handling and AMI evaluation. 

### `pipeline_part1.ipynb`

Data loading and preprocessing pipeline. Handles both molecular dynamics trajectories (XTC/DCD + PDB) and non-MD CSV datasets. Contains trajectory preparation functions, mean preprocessing, and data format conversion utilities. The output consists in tempral matrices (numpy arrays) which can be used as input for SHiP.

### `pipeline_part2.ipynb`

Main experiment execution notebook. Runs COMET/SHiP experiments on Cakmak benchmark datasets with balanced file selection, timeout handling, and ablation studies across matrix types (delta, delta+1std, stddv). Also includes tslearn competitor benchmarking (KShape, TS-KMeans).

### `pipeline_part3.ipynb`

Results aggregation and visualization. Loads and merges results from all benchmark runs (COMET, SHiP, Cakmak competitors, tslearn), computes speedup statistics, and generates comparison plots (runtime, NMI, ARI vs. number of timepoints/animals) by model (Calovi, Reynolds, Couzin).

### `md_part1.ipynb`

SHiP clustering on molecular dynamics protein trajectories. Loads MD trajectory data (e.g., TLL protease, F-peptide), computes distance matrices, runs SHiP clustering, and evaluates with Q scores.

### `md_part2.ipynb`

MD trajectory competitor comparison and visualization. Runs Resicon, CoMoDo, and GEOSTAS on protein trajectories, compares Q scores against SHiP/COMET results, and generates visualization plots.

## Project Structure

```
COMET-D73C/
├── redpandda.py                    # Original clustering pipeline (delta matrix + spectral clustering)
├── redpandda_general.py            # Core distance/delta matrix computation utilities
├── redpandda_without_ship.py       # Variant without SHiP dependency
├── distance_matrix.py              # Distance matrix computation + SHiP/Spectral/HDBSCAN clustering
├── clustering_functions.py         # COMET clustering workflow + timestep clustering
├── timestep_clustering.py          # Time-series clustering and change detection
├── geostas.py                      # GEOSTAS integration for protein structure clustering
├── postprocess_clusterings.py      # Noise point assignment and clustering post-processing
├── compare_clusterings.py          # Clustering evaluation metrics (AMI, ARI, NMI, Q)
├── cakmak_benchmarks.py            # SHiP vs ST-HDBSCAN on Cakmak benchmark datasets
├── cakmak_comparison_small_datasets.ipynb  # Full comparison with grid search optimization
├── run_competitors.py              # Cakmak ST competitors + COMET grid search
├── run_cakmak_competitors.py       # Lightweight ST competitor benchmarking (ST-Spectral, ST-DBSCAN, ST-HDBSCAN)
├── pipeline_part1.ipynb            # Data loading and preprocessing (MD + non-MD)
├── pipeline_part2.ipynb            # COMET/SHiP experiments + ablation + tslearn competitors
├── pipeline_part3.ipynb            # Results aggregation and visualization
├── md_part1.ipynb                  # SHiP on MD protein trajectories
├── md_part2.ipynb                  # MD competitor comparison (Resicon, CoMoDo, GEOSTAS)
├── CONSTANTS.py                    # Trajectory configurations for protein datasets
├── COMET_MD_env.yml                # Conda environment specification
├── requirements.txt                # pip dependencies
├── SHiP_dmat.yml                   # SHiP package environment
├── CoMoDo/                         # CoMoDo tools (legacy)
├── spatio-temporal-clustering-benchmark/  # ST clustering benchmark code
├── simulations/                    # Simulation data and results
├── paper_figures/                  # Generated publication figures
├── data/                           # Trajectory data files (.npz, .csv, .pdb)
└── *_example*.ipynb                # additional example notebooks
```

## Installation

### Conda environment

For running SHiP, use the environment in SHiP_dmat.yml: 

```bash
conda env create -f COMET_MD_env.yml
conda activate COMET_MD
```

For MD related functions (in particular preprocessing for SHiP), COMET_MD_env provides the necessary libraries:

```bash
conda env create -f COMET_MD_env.yml
conda activate COMET_MD
```

### pip dependencies

```bash
pip install -r requirements.txt
```

### R dependency

GEOSTAS clustering requires R's `bio3d` package:

```r
install.packages("bio3d", repos = "https://cloud.r-project.org")
```
