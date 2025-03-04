import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

import redpandda
from redpandda import *

import resicon
from resicon import *

import geostas
import mdtraj

import MDAnalysis as mda 

def mda_convert(input_xtc, input_pdb):
    print("the files")
    print(input_xtc)
    print(input_pdb)

    u = mda.Universe(input_pdb, input_xtc) # or use PDB or GRO instead of TPR
    print(u)
    print(u.trajectory)
    protein = u.select_atoms("all")
    print(protein.n_atoms)
    print(protein)
    dcd_file = input_xtc.split(".")[0] + ".dcd"
    #with mda.DCDTrajectoryFile('mytrajectory.dcd') as f:
    #    f.write(u)
    with mda.Writer(dcd_file, n_atoms=protein.atoms.n_atoms) as W:
        for ts in u.trajectory:
            print(ts)
            W.write(protein) 


def mda_convert_pdb(input_xtc, input_pdb):
    print("the files")
    print(input_xtc)
    print(input_pdb)

    u = mda.Universe(input_pdb, input_xtc) # or use PDB or GRO instead of TPR
    print(u)
    print(u.trajectory)
    protein = u.select_atoms("all")
    print(protein.n_atoms)
    print(protein)
    dcd_file = input_xtc.split(".")[0] + ".pdb"
    #with mda.DCDTrajectoryFile('mytrajectory.dcd') as f:
    #    f.write(u)
    with mda.Writer(dcd_file, n_atoms=protein.atoms.n_atoms, multiframe=True) as W:
        for ts in u.trajectory:
            print(ts)
            W.write(protein) 


def convert_folder(folder, input_pdb):
    import os 
    all_traj = []
    for file in os.listdir(folder):
    # check only text files
        if file.endswith('.xtc'):
            all_traj.append(folder + "/" + file)

    for input_xtc in all_traj:
        mda_convert(input_xtc, input_pdb)



def convert_folder_pdb(folder, input_pdb):
    import os 
    all_traj = []
    for file in os.listdir(folder):
    # check only text files
        if file.endswith('.xtc'):
            all_traj.append(folder + "/" + file)

    for input_xtc in all_traj:
        mda_convert_pdb(input_xtc, input_pdb)

