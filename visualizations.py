import matplotlib.pyplot as plt

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix
import sortedcontainers



def create_heatmap_from_data(data, title="Comparison between trajectories / clusterings", names=None, heatmap_label='Trajectory Nr.'):
    import numpy as np
    import matplotlib.pyplot as plt

    num_individuals = max(max(row[0] for row in data), max(row[1] for row in data)) + 1

    if names is None:
        names = [str(i) for i in range(num_individuals)]

    heatmap_data = np.zeros((num_individuals, num_individuals))

    for row in data:
        i, j, value = row
        heatmap_data[i][j] = value

    plt.figure(figsize=(8, 6))
    plt.imshow(heatmap_data, cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Values')
    plt.xlabel(heatmap_label)
    plt.ylabel(heatmap_label)
    plt.title(title)
    
    plt.xticks(ticks=np.arange(num_individuals), labels=names, rotation=90)
    plt.yticks(ticks=np.arange(num_individuals), labels=names)

    plt.show()



def get_trajectory_heatmap(clusterings, clusterings2, nmi = True, create_heatmap = True, title = "Comparison between trajectories / clusterings", names=None, heatmap_label='Trajectory Nr.'):
    import sklearn
    rand_array = []
    for i1, cl1 in  enumerate(clusterings):
        for i2, cl2 in enumerate(clusterings2):

            if not nmi:
                rand_array.append([i1,i2, sklearn.metrics.adjusted_rand_score(cl1, cl2)])
            else:
                rand_array.append([i1,i2, sklearn.metrics.normalized_mutual_info_score(cl1, cl2)])

    
    if create_heatmap:
        if nmi:
            create_heatmap_from_data(rand_array, title= title + "\n NMI", heatmap_label=heatmap_label, names=names)
        else:
            create_heatmap_from_data(rand_array, title= title + "\n adj. Rand", heatmap_label=heatmap_label, names=names)


    return rand_array    




def plot_residue_line(residues, labels):
    labels_set = sortedcontainers.SortedSet(labels)
    num_labels = len(labels_set)
    
    color_map = plt.cm.get_cmap('tab10', num_labels) 

    fig, ax = plt.subplots()

    for i in range(len(residues) - 1):
        x = [residues[i], residues[i + 1]]
        y = [0, 0]  
        label_color = color_map(labels_set.index(labels[i]))
        ax.plot(x, y, color=label_color, linewidth=14)  

    plt.axis('off')

    ax.annotate(f'{residues[0]}', (residues[0], 0), xytext=(5, -15), textcoords='offset points', ha='center', color='black')
    ax.annotate(f'{residues[-1]}', (residues[-1], 0), xytext=(-5, -15), textcoords='offset points', ha='center', color='black')


    plt.text((residues[0] + residues[-1]) / 2, -0.01, 'Residue number', ha='center')

    plt.show()



def plot_residue_line_plus_noise(residues, labels, noise_label = -1, title = None):
    labels_set = sortedcontainers.SortedSet(labels)
    num_labels = len(labels_set)
    
    color_map = plt.cm.get_cmap('tab10', num_labels) 

    fig, ax = plt.subplots()

    for i in range(len(residues) - 1):
        x = [residues[i], residues[i + 1]]
        y = [0, 0]  
        label_color = color_map(labels_set.index(labels[i]))
        if noise_label:
            if labels[i] == noise_label:
                label_color = "black"
        ax.plot(x, y, color=label_color, linewidth=14)  

    plt.axis('off')

    ax.annotate(f'{residues[0]}', (residues[0], 0), xytext=(5, -15), textcoords='offset points', ha='center', color='black')
    ax.annotate(f'{residues[-1]}', (residues[-1], 0), xytext=(-5, -15), textcoords='offset points', ha='center', color='black')


    plt.text((residues[0] + residues[-1]) / 2, -0.01, 'Residue number', ha='center')

    if title:
        plt.title(title)

    plt.show()


import numpy as np
import matplotlib.pyplot as plt
import sortedcontainers

import numpy as np
import matplotlib.pyplot as plt
import sortedcontainers

def multiple_line_plots(data, full_title="Residue clustering", 
                        orig_data=None, noise_value=-1, num_cols=1, plot_subtitle=True, plot_fulltitle=False):

    # Extract unique labels and sort them
    labels_set = sortedcontainers.SortedSet([y for item in data for y in item['labels']])
    
    # Define colormap
    num_labels = len(labels_set)
    color_map = plt.cm.get_cmap('tab20', num_labels)  

    # Determine grid layout
    num_plots = len(data)
    num_rows = (num_plots + num_cols - 1) // num_cols  

    # Create subplots
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, 2 * num_rows), gridspec_kw={'hspace': 0.2})
    
    # Ensure axs is iterable correctly
    if num_cols == 1:  
        axs = np.array(axs).reshape(-1)  # Force 1D array for single column
    else:
        axs = np.array(axs).reshape(num_rows, num_cols)  # Ensure 2D for multiple columns

    add_counter = 0
    for i, item_data in enumerate(data):
        row, col = divmod(i, num_cols)  # Compute correct row/column index
        
        # Correctly index subplot depending on num_cols
        ax = axs[row] if num_cols == 1 else axs[row, col]

        items = item_data['items']
        labels = item_data['labels']

        for k in range(len(items) - 1):
            x = [items[k], items[k + 1]]
            y = [0, 0]  
            label_color = color_map(labels_set.index(labels[k]))

            # Handle noise coloring
            if orig_data is not None and orig_data[add_counter][k] == noise_value:
                label_color = 'black'

            ax.plot(x, y, color=label_color, linewidth=20)  
            ax.axis('off')  

        # Annotate residue numbers
        ax.annotate(f'{items[0]}', (items[0], 0), xytext=(5, -26), textcoords='offset points', ha='center', color='black')
        ax.annotate(f'{items[-1]}', (items[-1], 0), xytext=(-5, -26), textcoords='offset points', ha='center', color='black')
        ax.text((items[0] + items[-1]) / 2, -0.03, 'Residue number', ha='center')

        if plot_subtitle:
            ax.set_title(item_data['title'], fontsize=10)
        add_counter += 1

    # Remove unused subplots
    if num_cols > 1:
        for j in range(i + 1, num_rows * num_cols):
            row, col = divmod(j, num_cols)
            fig.delaxes(axs[row, col])  

    # Set full title and display plot
    if plot_fulltitle:
        plt.suptitle(full_title)
    plt.tight_layout()
    plt.show()



def rearrange_labels_multi(*label_lists):
    """
    Rearrange multiple lists of labels to match the first list using the Hungarian (Munkres) algorithm.
    """
    num_lists = len(label_lists)
    num_labels = max(max(labels) for labels in label_lists) + 1
    
    agg_cm = np.zeros((num_labels, num_labels))
    
    for labels1 in label_lists:
        for labels2 in label_lists:
            cm = confusion_matrix(labels1, labels2, labels=range(num_labels))
            agg_cm += cm
    
    row_ind, col_ind = linear_sum_assignment(-agg_cm)  
    
    rearranged_label_lists = [[col_ind[label] for label in labels] for labels in label_lists]
    
    return rearranged_label_lists


def max_overlap_matching(timestep_clusters):
    """
    Maximize the overlap between clustering labels across consecutive time steps using the Hungarian algorithm.

    Args:
    timestep_clusters (list): List of dictionaries containing 'start', 'end', and 'clustering' arrays.

    Returns:
    List: A list of dictionaries with updated clustering labels after applying the Hungarian algorithm.
    """
    updated_clusters = []

    # Iterate through each pair of consecutive time steps
    for i in range(len(timestep_clusters) - 1):
        # Get the current and next time step clusters
        curr_start, curr_end, curr_clustering = timestep_clusters[i]['start'], timestep_clusters[i]['end'], timestep_clusters[i]['clustering']
        next_start, next_end, next_clustering = timestep_clusters[i + 1]['start'], timestep_clusters[i + 1]['end'], timestep_clusters[i + 1]['clustering']

        # Get the number of clusters in the current and next time step
        num_curr_clusters = len(np.unique(curr_clustering))
        num_next_clusters = len(np.unique(next_clustering))

        # Create a cost matrix where the rows are current clusters and columns are next clusters
        cost_matrix = np.zeros((num_curr_clusters, num_next_clusters), dtype=int)

        # Populate the cost matrix with the number of overlapping objects between each pair of clusters
        for curr_label in range(num_curr_clusters):
            for next_label in range(num_next_clusters):
                # Calculate the overlap (number of objects with the same label in both clusters)
                overlap = np.sum((curr_clustering == curr_label) & (next_clustering == next_label))
                cost_matrix[curr_label, next_label] = -overlap  # Negative because we want to maximize overlap

        # Use the Hungarian algorithm to find the optimal assignment
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # Create a mapping from the current clusters to the next clusters
        label_mapping = dict(zip(row_ind, col_ind))

        # Update the clustering labels for the next time step based on the mapping
        next_clustering_updated = np.copy(next_clustering)
        for curr_label, next_label in label_mapping.items():
            next_clustering_updated[next_clustering == next_label] = curr_label

        # Store the updated clustering in the result list
        updated_clusters.append({
            'start': curr_start,
            'end': curr_end,
            'clustering': curr_clustering
        })

        # Update the next step with the new clustering labels
        timestep_clusters[i + 1]['clustering'] = next_clustering_updated

    # Add the last cluster (as it doesn't have a next step to compare with)
    updated_clusters.append(timestep_clusters[-1])

    return updated_clusters



def create_list_of_dicts(labels, items=None, titles=None):
    if items is None:
        items = []
        for label in labels:

            items.append(list(range(len(label))))

    if titles is None:
        titles = []
        for il, label in enumerate(labels):
            titles.append("plot nr. " + str(il))


    list_of_dicts = []

    for i in range(len(labels)):
        title = titles[i]
        item = items[i]
        label = labels[i]
        try:
            data_dict = {'title': title, 'items': item, 'labels': label.tolist()}
        except:
            data_dict = {'title': title, 'items': item, 'labels': label}
        list_of_dicts.append(data_dict)
    return list_of_dicts


def reassign_labels(cluster_labels1, cluster_labels2):
    unique_labels1 = np.unique(cluster_labels1)
    unique_labels2 = np.unique(cluster_labels2)
    
    num_clusters1 = len(unique_labels1)
    num_clusters2 = len(unique_labels2)
    
    cost_matrix = np.zeros((num_clusters1, num_clusters2))
    for i in range(num_clusters1):
        for j in range(num_clusters2):
            common_elements = np.sum((cluster_labels1 == unique_labels1[i]) & (cluster_labels2 == unique_labels2[j]))
            cost_matrix[i, j] = -common_elements 
    
    # Use the Hungarian (Munkres) algorithm to find the optimal assignment
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    label_mapping = {}
    for i, j in zip(row_ind, col_ind):
        if cost_matrix[i, j] != -np.inf: 
            label_mapping[unique_labels2[j]] = unique_labels1[i]
    
    reassigned_labels2 = [label_mapping[label] if label in label_mapping else label for label in cluster_labels2]
    
    return cluster_labels1, reassigned_labels2



def iterate_rearrange_labels(labels_lists):
    print(labels_lists)
    try:
     start_label = labels_lists[0].tolist()
    except:
     start_label = labels_lists[0]
       

    rearranged_lists = []
    rearranged_lists.append(start_label)

    for cluster_list in labels_lists[1:]:

          if isinstance(cluster_list, list):
               cluster_list = cluster_list[0]

          new_cluster_list = reassign_labels(start_label, cluster_list)
          print(new_cluster_list)
          rearranged_lists.append(new_cluster_list[-1])

    return rearranged_lists



def line_plot_workflow(data, titles = "", full_title = "residue line plots", rearrange = True, hdb_scan_noise=False, num_cols=1):
    from copy import deepcopy
    data_final = deepcopy(data)
    if rearrange:
        data_final = iterate_rearrange_labels(data)
    dicts = create_list_of_dicts(data_final, titles = titles)

    if not hdb_scan_noise:
        multiple_line_plots(dicts, full_title = full_title, num_cols=num_cols)
    else:
        multiple_line_plots(dicts, full_title = full_title, orig_data = data, num_cols=num_cols)


def find_noise_resiudes(clusters, noise_nr=-1):
    indices = [i for i, x in enumerate(clusters) if x == noise_nr]
    return indices

