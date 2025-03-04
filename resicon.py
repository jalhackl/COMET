import numpy as np
import mdtraj as md
import manipulate_trajectory as mt
import math
import CONSTANTS
from itertools import combinations
from sklearn.cluster import SpectralClustering

from functools import wraps
from time import time

def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print('%2.4f' % (te-ts))
        #print('func:%r took: %2.4f sec' % (f.__name__, te-ts))
        return result
    return wrap

@timing
def get_trajectory(protein_file, prot_topology_file, protein_folder):

    if prot_topology_file != None:
      traj = md.load(CONSTANTS.path_prefix + protein_folder + protein_file, top=CONSTANTS.path_prefix + protein_folder + prot_topology_file)
    else:
      traj = md.load(CONSTANTS.path_prefix + protein_folder + protein_file)
    res_w_ca_indices = np.array([atom.residue.index for atom in traj.topology.atoms if atom.name == 'CA'])
    atoms_to_keep = np.array([atom.index for atom in traj.topology.atoms if atom.residue.index in res_w_ca_indices])
    #print('Residue count pre removing residues w/o CA atoms: ' + str(traj.n_residues))
    traj = traj.atom_slice(atoms_to_keep)
    # for atom in traj.topology.residues:
    #   print(str(atom) + ' id: ' + str(atom.residue.index))
    #print('Residue count after: ' + str(traj.n_residues))
    return traj

def get_centers_of_sidechains(traj: md.Trajectory, only_CA=False):
    '''
    traj: Trajectory of current protein
    Returns an array containing the geometrical centers of the sidechains of each residue for each frame.
    Shape: (n_residues, n_frames, 3)
    '''
    geo_centers = np.zeros((traj.n_residues, traj.n_frames, 3))
    for res in traj.topology.residues:
      print("Resicon residue:")
      print(res.name)
      print(res.index)

      if only_CA == False:
        if res.name == 'ALA' or res.name == 'GLY':
          center_atom_name = 'CA' if res.name == 'ALA' else 'C' # Glycin has 2 Carbonatoms: one CA and one plain C
          atom_id = [atom.index for atom in res.atoms if atom.name == center_atom_name]
          if len(atom_id) != 1:
            raise ValueError('More than 1 or none alpha or beta carbon detected in residue ' + str(res.index))
          geo_centers[res.index] = np.squeeze(traj.atom_slice(atom_id).xyz)  # Squeeze to turn array of shape (n_frames,1,3) into (n_frames,3)
          continue
      # Extract atoms that are in the sidechain of a reidue
      res_sidechain_coords = traj.atom_slice([atom.index for atom in res.atoms if atom.is_sidechain]).xyz  # Returns array of shape (n_frames,x_sc_atoms,3)
      geo_centers[res.index] = np.mean(res_sidechain_coords,axis=1)
    return geo_centers


def possible_combinations(x: int):
  return x*x - np.sum(range(x+1))

def check_side_condition(frame, dist_sc, res_pair, res_pairs_ca, dists_ca):
  '''
  frame: The current frame
  dist_sc: The current distance between the residue pair
  res_pair: The residue pair
  res_pairs_ca: The residue pairs |res_pairs| x 2 of the alpha carbon computation
  dists_ca: The distances |frames| x |res_pairs| of the alpha carbon computation

  Returns True, if dist_ca - dist_sc >= 0.75A and False otherwise
  '''
  for pair_idx, pair in enumerate(res_pairs_ca):
    if np.array_equal(pair,res_pair):
      if dists_ca[frame][pair_idx] - dist_sc >= 0.075:
        return True
      return False
  return False

#@timing
def get_elements(residue_pairs: np.ndarray, n_residues: int, traj_cas: md.Trajectory):
  '''
  residue_pairs: |res_pairs_with_contact| x 2 contains pairs of residue indices that are in contact
  n_residues: Amount of residues in protein
  traj_cas: Trajectory containing only alpha carbon atoms

  Return a dictionary res_to_elem: key = residue_idx, value = carbon alpha atoms of element
  '''
  res_indices = np.unique(residue_pairs.flatten())
  res_to_elem = dict()

  for res_idx in res_indices:
    # First two and last two residues can not be part of an element
    if res_idx < 2 or res_idx >= n_residues-1-2:   # res_idx start at 0
      continue
    else:
      ca_res_indices = [res_idx-2, res_idx-1, res_idx, res_idx+1, res_idx+2]
      ca_indices = np.array([ca.index for ca in traj_cas.topology.atoms if ca.residue.index in ca_res_indices])
      # If less than 5 cas found, do not add element
      if len(ca_indices) != 5:
          print('Less than 5 alpha carbons in neighborhood -> continue')
          continue
      res_to_elem[res_idx] = ca_indices
  
  return res_to_elem

def calculate_contact_value(traj_ca: md.Trajectory, res_to_elem: dict, res1_id: int, res2_id: int):
  elem1_ca_ids = res_to_elem.get(res1_id)
  elem2_ca_ids = res_to_elem.get(res2_id)

  if elem1_ca_ids is None or elem2_ca_ids is None:
    return 0
  
  combined_ca_ids = np.sort(np.unique(np.concatenate((elem1_ca_ids, elem2_ca_ids))))
  traj_elems = traj_ca.atom_slice(combined_ca_ids).center_coordinates()

  traj_elems = traj_elems[::int(traj_elems.n_frames/1000)] # TODO: Only in here to make calculation faster
  reduced_frames = traj_elems.n_frames
  max_rmsd = 0
  for frame in range(reduced_frames-1):
    current_max = np.max(md.rmsd(traj_elems[frame:reduced_frames], traj_elems, frame, precentered=True))
    if current_max > max_rmsd:
      max_rmsd = current_max
  
  return max_rmsd


#@timing
def calculate_contact_matrix(traj_ca, res_contact_pairs, res_to_elem, n_residues):

  contact_matrix = np.full((n_residues,n_residues), None)
  max_rmsds = list()

  for res1_id in range(n_residues):
    for res2_id in range(res1_id, n_residues):  # Improve runtime by calculate (res1,res2) and (res2, res1) in one step
      if abs(res1_id-res2_id) <= 1:
        contact_matrix[res1_id,res2_id] = 1
        contact_matrix[res2_id,res1_id] = 1
        continue
      # If residue pair in contact
      if np.any((res_contact_pairs == [res1_id, res2_id]).all(axis=1)):
        max_rmsd = calculate_contact_value(traj_ca, res_to_elem, res1_id, res2_id)
        contact_matrix[res1_id, res2_id] = max_rmsd
        contact_matrix[res2_id, res1_id] = max_rmsd
        max_rmsds.append(max_rmsd)
      else:
        contact_matrix[res1_id, res2_id] = 0
        contact_matrix[res2_id,res1_id] = 0
  
  max_rmsds = np.array(max_rmsds)
  max_rmsds = max_rmsds[max_rmsds != 0]
  
  alpha = np.mean(max_rmsds)
  beta = 1/np.std(max_rmsds)
  l = lambda x: x if x == 0 or x == 1 else 1/(1 + math.exp((x-alpha)/beta))
  contact_matrix = np.array(list(map(l,contact_matrix.flatten()))).reshape(n_residues,n_residues)
  
  return contact_matrix

def compute_sc_dists(res_pairs: np.ndarray, geo_centers: np.ndarray):
    '''
    res_pairs: Pairs (|res_pairs| x 2) of residues to calculate the distances between their sidechain's geo center for 
    geo_centers: Array (n_residues, n_frames, 3) containing the geometrical centers of the sidechains of each residue for each frame

    Returns distances in the shape: (|frames| x |res_pairs|)
    '''
    frame_count = len(geo_centers[0])
    distances = np.zeros((len(res_pairs),frame_count))
    for res_pair_id, res_pair in enumerate(res_pairs):
      geo_centers_pair = geo_centers[list(res_pair)]
      distances[res_pair_id] = np.linalg.norm(geo_centers_pair[0] - geo_centers_pair[1], axis=1)
    return distances.transpose()


#@timing
def get_residue_contacts(traj: md.Trajectory, geo_centers: np.ndarray):
    '''
    traj: Trajectory of current protein
    geo_centers: Array (n_residues, n_frames, 3) containing the geometrical centers of the sidechains of each residue for each frame
    '''
    n_residues = traj.topology.n_residues

    res_pair_candidates = np.array(list(combinations(range(n_residues), 2)))
    res_pair_candidates = [(x,y) for (x,y) in res_pair_candidates if x+1 != y]
    
    # dists: distances |frames| x |res_pairs|,  res_pairs: |residue_pairs| x 2
    dists_ca, res_pairs_ca = md.compute_contacts(traj,contacts=res_pair_candidates,scheme='ca')
    
    # Only keep residue pairs that fulfill contact condition
    # First sufficient (hinreichende) Condition: d_ca <= 6.5A
    idx_list_res_contact_ca = list()
    for idx, dists in enumerate(dists_ca.T):
       if np.any(dists <= 0.65):    #1A = 0.1nm
        idx_list_res_contact_ca.append(idx)
    contact_pairs_ca = res_pairs_ca[np.ix_(idx_list_res_contact_ca)]
    #contact_dists_ca = dists_ca[np.ix_(range(len(dists_ca)),idx_list_res_contact_ca)]
    
    # Second sufficient Condition: d_sc <= 8A    AND   d_ca - d_sc >= 0.75A
    res_pairs_sc = np.delete(res_pairs_ca, idx_list_res_contact_ca, axis=0)
    dists_sc = compute_sc_dists(res_pairs_sc,geo_centers)

    idx_list_res_contact_sc = list()
    for idx, dists in enumerate(dists_sc.T):
       # Get resiude index of current residue pair in res_pairs_ca
       pair_idx_ca = np.where((res_pairs_ca == res_pairs_sc[idx]).all(axis=1))[0]
       if len(pair_idx_ca) == 0: continue
       else: pair_idx_ca = pair_idx_ca[0]

       for frame, dist in enumerate(dists):
        if dist <= 0.8 and dists_ca[frame][pair_idx_ca] - dist >= 0.075:
          idx_list_res_contact_sc.append(idx)
          break
    contact_pairs_sc = res_pairs_sc[np.ix_(idx_list_res_contact_sc)]
    #contact_dists_sc = dists_sc[np.ix_(range(len(dists_sc)),idx_list_res_contact_sc)]

    return np.concatenate((contact_pairs_ca, contact_pairs_sc))

@timing
def resicon(protein_file: str, topology_file: str, folder: str, k_clusters: int, only_CA=False):
  traj = get_trajectory(protein_file,topology_file,folder)
  geo_centers = get_centers_of_sidechains(traj, only_CA=only_CA)
  res_contacts = get_residue_contacts(traj, geo_centers)
  #print(res_contacts.shape)
  #print('Possible combis: ' + str(possible_combinations(traj.n_residues)))

  traj_ca = mt.restrict_protein_to_elements(traj,'CA')
  #print('CA count: ' + str(traj_ca.n_atoms))
  res_to_elem = get_elements(res_contacts, traj.n_residues, traj_ca)
  contact_matrix = calculate_contact_matrix(traj_ca, res_contacts, res_to_elem, traj.n_residues)

  clustering = SpectralClustering(n_clusters=k_clusters, assign_labels='discretize', random_state=0, affinity='precomputed')
  
  return clustering.fit(contact_matrix).labels_

#proteins = CONSTANTS.r1

#for prot_info in proteins:
#  print(prot_info[0])
#  resicon(prot_info[0],prot_info[1],prot_info[2],prot_info[4])

#for prot_info in CONSTANTS.abeta:
#  print(prot_info[0])
#  resicon(prot_info[0],prot_info[1],prot_info[2],prot_info[4])
