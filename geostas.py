from rpy2 import robjects
from rpy2.robjects.packages import importr
from rpy2.robjects.vectors import ListVector
from rpy2.robjects import pandas2ri



def compute_geostas_clusters_multipdb(filename, install=False, only_CA=True, k=3):
    if install == True:
        utils = importr('utils')
        utils.install_packages('bio3d', repos="https://cloud.r-project.org")

    bio3d = importr('bio3d')
    bio3d.read_pdb = bio3d.read_pdb
    geostas = bio3d.geostas

    base = importr('base')

    pdbfile = bio3d.read_pdb(filename, multi=True)
    if only_CA:
        pdbfile = bio3d.trim(pdbfile, atom='CA')


    geostas_result = geostas(pdbfile, k=k, fit=True)

    l = geostas_result
    d = dict(l.items())

    clustering = pandas2ri.PandasDataFrame(d["grps"]).to_numpy().flatten()

    return clustering



def compute_geostas_clusters_dcd(filename, pdb_file=None, install=False, only_CA=True, k=3):
    if install == True:
        utils = importr('utils')
        utils.install_packages('bio3d', repos="https://cloud.r-project.org")

    bio3d = importr('bio3d')
    bio3d.read_dcd = bio3d.read_dcd
    geostas = bio3d.geostas

    base = importr('base')




    trajectory_file = bio3d.read_dcd(filename)


    if only_CA:
        if pdb_file:
            # Read PDB file to get atom information
            pdb = bio3d.read_pdb(pdb_file)

            # Select only CA atoms
            ca_selection = bio3d.atom_select(pdb, "calpha")

            # Apply CA selection to DCD trajectory
            trajectory_file = bio3d.trim(trajectory_file, ca_selection)

    geostas_result = geostas(trajectory_file, k=k, fit=True)

    l = geostas_result
    d = dict(l.items())

    clustering = pandas2ri.PandasDataFrame(d["grps"]).to_numpy().flatten()

    return clustering




   

