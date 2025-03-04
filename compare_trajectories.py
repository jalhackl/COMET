def create_heatmap_from_data(data, title= "Comparison between trajectories / clusterings"):
    import numpy as np
    import matplotlib.pyplot as plt

    num_individuals = max(max(row[0] for row in data), max(row[1] for row in data)) + 1

    # Initialize a 2D array
    heatmap_data = np.zeros((num_individuals, num_individuals))

    # Fill in the values from your data
    for row in data:
        i, j, value = row
        heatmap_data[i][j] = value

    # Plotting the heatmap
    plt.figure(figsize=(8, 6))
    plt.imshow(heatmap_data, cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Values')
    plt.xlabel('Trajectory Nr.')
    plt.ylabel('Trajectory Nr.')
    plt.title(title)
    plt.show()


def get_trajectory_heatmap(clusterings, clusterings2, nmi = True, create_heatmap = True, title = "Comparison between trajectories / clusterings"):
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
            create_heatmap_from_data(rand_array, title= title + "\n NMI")
        else:
            create_heatmap_from_data(rand_array, title= title + "\n adj. Rand")


    return rand_array