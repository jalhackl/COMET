import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import diptest


def dist_heatmaps_for_matrices(input_matrix,title="Heatmap of pair distributions across trajectory",plot_kde=True,check_unimodality=True,
                               manipulate_according_to_modality=True, modality_alpha=0.05, plot_p_value=False,
                               manipulate_and_return_data=False, return_mean=True, multimodal_penalty=True, penalty=10):


    # Transpose the array to (1, 2, 0) ordering
    transposed_array = np.transpose(input_matrix, (1, 2, 0))

    # Compute global statistics
    global_min_x = np.min(transposed_array)
    global_max_x = np.max(transposed_array)
    global_min_mean = np.min(np.mean(transposed_array, axis=2))
    global_max_mean = np.max(np.mean(transposed_array, axis=2))

    # Compute mean values for color scaling
    mean_values = np.mean(transposed_array, axis=2)

    # Create a grid of subplots
    fig, axes = plt.subplots(transposed_array.shape[0], transposed_array.shape[0], figsize=(20, 20))

    if manipulate_and_return_data:
        output_matrix = np.zeros((transposed_array.shape[0], transposed_array.shape[1]))

    for row in range(transposed_array.shape[0]):
        for col in range(transposed_array.shape[0]):
            ax = axes[row, col]
            
            # Extract the values for the (row, col) position across all distance matrices
            cell_values = transposed_array[row, col, :]  # All values for the current cell
            
            # Remove x and y labels and ticks
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_ylabel('')  

            # KDE plot to show the distribution
            if plot_kde:
                sns.kdeplot(cell_values, ax=ax, color='skyblue', linewidth=3.5)
            # otherwise the values across the trajectory are plotted
            else:
                sns.lineplot(cell_values, ax=ax, color='skyblue', linewidth=2.5)

            # Add a background color using Rectangle based on mean values
            color = plt.cm.coolwarm((mean_values[row, col] - global_min_mean) / (global_max_mean - global_min_mean))

            # Compute the dip test for unimodality
            if check_unimodality:
                dip_statistic, p_value = diptest.diptest(cell_values)

                if manipulate_according_to_modality:
                    # If the distribution is NOT unimodal (p < modality_alpha), color the background dark grey
                    if p_value < modality_alpha:
                        color = '#555555'  # Dark grey

                # Display the p-value in the center of the cell
                if plot_p_value:
                    ax.text(0.5, 0.5, f"p={p_value:.3f}", color='black', fontsize=10, ha='center', va='center', transform=ax.transAxes)


            # color boxes
            rect = Rectangle((0, 0), 1, 1, color=color, transform=ax.transAxes, zorder=1)
            ax.add_patch(rect)

            # Add grid lines for clarity
            ax.grid(visible=True, linestyle='--', linewidth=0.5, color='gray')

            # Set consistent x-axis range
            ax.set_xlim(global_min_x, global_max_x)


            if manipulate_and_return_data:
                if return_mean:
                    new_value = np.mean(cell_values)
                if multimodal_penalty and check_unimodality:
                    if p_value < modality_alpha:
                        new_value = global_max_x * penalty

                output_matrix[row, col] = new_value


    # Adjust layout for clarity and reduce spacing
    plt.tight_layout()

    plt.subplots_adjust(wspace=0.02, hspace=0.02)

    fig.suptitle(title, fontsize=16, y=1.02) 
    plt.show()

    if manipulate_and_return_data:
        return output_matrix