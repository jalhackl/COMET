from rpy2 import robjects
from rpy2.robjects.packages import importr
from rpy2.robjects.vectors import ListVector
from rpy2.robjects import pandas2ri



def compute_geostas_clusters_multipdb(filename, install=False):
    if install == True:
        utils = importr('utils')
        utils.install_packages('bio3d', repos="https://cloud.r-project.org")

    bio3d = importr('bio3d')
    bio3d.read_pdb = bio3d.read_pdb
    geostas = bio3d.geostas

    base = importr('base')

    pdbfile = bio3d.read_pdb(filename, multi=True)

    geostas_result = geostas(pdbfile, fit=True)

    l = geostas_result
    d = dict(l.items())

    clustering = pandas2ri.PandasDataFrame(d["grps"]).to_numpy().flatten()

    return clustering



def compute_geostas_clusters_dcd(filename, install=False):
    if install == True:
        utils = importr('utils')
        utils.install_packages('bio3d', repos="https://cloud.r-project.org")

    bio3d = importr('bio3d')
    bio3d.read_dcd = bio3d.read_dcd
    geostas = bio3d.geostas

    base = importr('base')

    pdbfile = bio3d.read_dcd(filename)

    geostas_result = geostas(pdbfile, fit=True)

    l = geostas_result
    d = dict(l.items())

    clustering = pandas2ri.PandasDataFrame(d["grps"]).to_numpy().flatten()

    return clustering




   

