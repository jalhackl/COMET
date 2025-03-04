from matplotlib.cbook import flatten
import numpy as np
import mdtraj as md

def restrict_protein_to_elements(traj, elementName) -> md.Trajectory:
    atoms_to_keep = [a.index for a in traj.topology.atoms if a.name == elementName]
    return traj.atom_slice(atoms_to_keep)

def get_coords_of_interest(coords: np.ndarray, frames: list, atoms: list) -> np.ndarray:
  return coords[np.ix_(frames, atoms)]

def get_all_coords_at_frame(coords: np.ndarray, frame: int) -> np.ndarray:
  return coords[frame]

def get_coords_of_atom_at_frame(coords: np.ndarray, frame: int, atom: int) -> np.ndarray:
  return coords[frame][atom]

def traj_to_list(coords: np.ndarray) -> list:
  flattened_data = []
  for (frame_idx,frame) in enumerate(coords):
    for (atom_idx,atom) in enumerate(frame):
      flattened_data.append([atom_idx, frame_idx, atom[0], atom[1], atom[2]])
  return flattened_data


def thin_trajectory(trajectory, thinning_factor):
  trajectory = trajectory[:, 0:trajectory.shape[1]:int(trajectory.shape[1] / thinning_factor), :]
  return trajectory
