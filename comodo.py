from rpy2 import robjects
from rpy2.robjects.packages import importr
from rpy2.robjects.vectors import ListVector
from rpy2.robjects import pandas2ri
import rpy2.robjects as ro
import pandas as pd
import re
import redpandda
from redpandda import *
import os

def parse_comodo_domains(text):
    domains = {}
    current_domain = None
    lines = text.split('\n')
    for line in lines:
        if line.startswith('Nodes in Domain'):
            current_domain = int(line.split(':')[0][-1])
        elif current_domain is not None and line.startswith('nodes'):
            nodes_range_str = re.findall(r'\d+-\d+|\d+', line)
            nodes_list = []
            for item in nodes_range_str:
                if '-' in item:
                    start, end = map(int, item.split('-'))
                    nodes_list.extend(range(start, end + 1))
                else:
                    nodes_list.append(int(item))
            domains[current_domain] = nodes_list
    return domains


def compute_gnm_modes(filename, output_file="new_cov.cov", write_file=False, install=False):
    if install == True:
        utils = importr('utils')
        utils.install_packages('bio3d', repos="https://cloud.r-project.org")

    bio3d = importr('bio3d')

    #cov = importr('cov')

    bio3d.read_pdb = bio3d.read_pdb
    gnm = bio3d.gnm
    #covnma = bio3d.cov

    base = importr('base')

    covnma = ro.r('cov.nma')

    pdbfile = bio3d.read_pdb(filename, multi=True)

    modes  = gnm(pdbfile, full=True)

    nma_object = covnma(modes)


    #geostas_result = geostas(pdbfile, fit=True)

    #l = geostas_result
    #d = dict(l.items())

    #clustering = pandas2ri.PandasDataFrame(d["grps"]).to_numpy().flatten()

    r_code = """
n <- nrow(covariance_matrix)

indices <- c()
cov_values <- c()

# Iterate over the upper triangular part of the covariance matrix
for (i in 1:n) {
  for (j in 1:n) {
    # extract iindex of the first atom, the index of the second atom, and the covariance value
    index1 <- i
    index2 <- j
    cov <- covariance_matrix[i, j]

    indices <- c(indices, index1, index2)
    cov_values <- c(cov_values, cov)
  }
}

# Combine the lists 
cov_df <- data.frame(Atom_Index_1 = indices[seq(1, length(indices), by=2)],
                     Atom_Index_2 = indices[seq(2, length(indices), by=2)],
                     Covariance = cov_values)

# Return the resulting data frame
cov_df
"""

    covariance_matrix = nma_object
    ro.globalenv['covariance_matrix'] = covariance_matrix


    result = ro.r(r_code)

    cov_df = pd.DataFrame(result)#.rx(True)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    if write_file:
        cov_df = cov_df.T
        cov_df[[0,1]] = cov_df[[0,1]].astype(int)

        with open(output_file, 'w') as f:
            df_string = cov_df.to_string(header=False, index=False, justify="left")
            f.write(df_string)


    return cov_df

#alternative, one could compute the covariances from the full set...


def full_covariance_matrix(prot_info, output_file="new_cov.cov", write_file=False):
        
    traj = preprocessing_return_traj(prot_info,-1,None)

    #print("TRAJDATA")
    #print(traj_data)


    positions = traj[0].xyz



    #mean position
    mean_position = np.mean(positions, axis=0)

    deviation = positions - mean_position

    #covariance matrix
    covariance_matrix = np.einsum('...ij,...kj->...ik', deviation, deviation) / len(traj[0])

    #final covariance matrix
    covariance_matrix = np.sum(covariance_matrix, axis=0)

    num_atoms = covariance_matrix.shape[0]

    print("orig. covariance_matrix")
    print(covariance_matrix)

    cov_data = []

    for i in range(num_atoms):
        for j in range(num_atoms):
            index1 = i
            index2 = j
            covariance = covariance_matrix[i, j]
            
            cov_data.append((index1+1, index2+1, covariance))

    cov_df = pd.DataFrame(cov_data, columns=["Atom_Index_1", "Atom_Index_2", "Covariance"])

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    if write_file:
        
        cov_df[["Atom_Index_1", "Atom_Index_2"]] = cov_df[["Atom_Index_1", "Atom_Index_2"]].astype(int)

        with open(output_file, 'w') as f:
            df_string = cov_df.to_string(header=False, index=False, justify="left")
            f.write(df_string)
    print("Covariance dataframe:")
    print(cov_df)

    return cov_df



def run_comodo_clusterer(cov_file):
    import subprocess

    command = ['./CoMoDo/src/DomainClusterer', str(cov_file), '-c', '0', '-v']

    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    newtext = result.stdout.decode()

    domains = parse_comodo_domains(newtext)

    return domains


def full_comodo_clustering(filename, elastic_network=True):
    try:
        cov_file = filename.split(".")[0] + ".cov"
    except:
        cov_file = filename[0].split(".")[0] + ".cov"

    if elastic_network:
        compute_gnm_modes(filename, output_file=cov_file, write_file=True, install=False)
    else:
        full_covariance_matrix(filename, output_file=cov_file, write_file=True)

    domains = run_comodo_clusterer(cov_file)
    print(domains)

    max_node = max(max(nodes) for nodes in domains.values())
    clustering = [-1] * max_node  
    for domain, nodes_list in domains.items():
        for node in nodes_list:
            clustering[node - 1] = domain

    return clustering

    

