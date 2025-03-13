def fast_kde(data, bins=50):
    """ Approximate KDE using histogram-based density estimation for speed. """
    hist, bin_edges = np.histogram(data, bins=bins, density=True)
    return bin_edges[:-1], hist

def process_cell(ax, row, col, transposed_array, mean_values, global_min_mean, global_max_mean, 
                 plot_kde, check_unimodality, modality_alpha, plot_p_value):
    
    cell_values = transposed_array[row, col, :]
    mean_dist = mean_values[row, col]
    
    # Remove axis labels for a clean heatmap
    ax.set_xticks([]) 
    ax.set_yticks([])
    
    # Faster KDE / Lineplot
    if plot_kde:
        x_vals, y_vals = fast_kde(cell_values)  # Approximate KDE
        ax.fill_between(x_vals, y_vals, color='skyblue', alpha=0.6)
    else:
        ax.plot(cell_values, color='skyblue', linewidth=2)
    
    # Background color based on mean
    color = plt.cm.coolwarm((mean_dist - global_min_mean) / (global_max_mean - global_min_mean))
    
    # Unimodality Test (Optional)
    if check_unimodality:
        dip_statistic, p_value = diptest.diptest(cell_values)
        if p_value < modality_alpha:
            color = '#555555'  # Dark grey for multimodal
        if plot_p_value:
            ax.text(0.5, 0.5, f"p={p_value:.3f}", color='black', fontsize=8, ha='center', va='center', transform=ax.transAxes)

    # Apply background color
    rect = Rectangle((0, 0), 1, 1, color=color, transform=ax.transAxes, zorder=1)
    ax.add_patch(rect)

def fast_dist_heatmaps(input_matrix, title="Fast Heatmap of Pairwise Distances", plot_kde=True, 
                       check_unimodality=True, modality_alpha=0.05, plot_p_value=False, n_jobs=-1):
    
    transposed_array = np.transpose(input_matrix, (1, 2, 0))
    N = transposed_array.shape[0]
    
    # Precompute mean values for color scaling
    mean_values = np.mean(transposed_array, axis=2)
    global_min_mean, global_max_mean = np.min(mean_values), np.max(mean_values)
    
    # Create a grid of subplots
    fig, axes = plt.subplots(N, N, figsize=(N * 2, N * 2))

    # Flatten axes for easy parallel mapping
    axes_flat = axes.flatten()

    # Parallel Processing for Subplots
    Parallel(n_jobs=n_jobs)(
        delayed(process_cell)(axes_flat[row * N + col], row, col, transposed_array, mean_values, 
                              global_min_mean, global_max_mean, plot_kde, check_unimodality, 
                              modality_alpha, plot_p_value) 
        for row in range(N) for col in range(N)
    )

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()