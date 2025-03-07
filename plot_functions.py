import numpy as np
import plotly.graph_objs as go
import matplotlib.pyplot as plt


def plot_traj_labels_plotly(trajectories, num_particles_per_group=None):
    num_timesteps = len(trajectories[0])
    num_particles = len(trajectories)
    groups_list = np.array(trajectories)[:,:,-1]
    num_groups = len(np.unique(np.array(groups_list).flatten()))

    num_groups = len(num_particles_per_group)

    frames = []
    for t in range(num_timesteps):
        frame_data = []
        particle_index = 0
        for group_index in range(num_groups):
            num_particles = num_particles_per_group[group_index][0]
            xs = [trajectories[particle_index + p][t][0] for p in range(num_particles)]
            ys = [trajectories[particle_index + p][t][1] for p in range(num_particles)]
            zs = [trajectories[particle_index + p][t][2] for p in range(num_particles)]
            frame_data.append(go.Scatter3d(
                x=xs, y=ys, z=zs,
                mode='markers',
                marker=dict(size=5),
                name=f'Group {group_index + 1}'
            ))
            particle_index += num_particles
        frames.append(go.Frame(data=frame_data, name=str(t)))

    initial_data = []
    particle_index = 0
    for group_index in range(num_groups):
        num_particles = num_particles_per_group[group_index][0]
        xs = [trajectories[particle_index + p][0][0] for p in range(num_particles)]
        ys = [trajectories[particle_index + p][0][1] for p in range(num_particles)]
        zs = [trajectories[particle_index + p][0][2] for p in range(num_particles)]
        initial_data.append(go.Scatter3d(
            x=xs, y=ys, z=zs,
            mode='markers',
            marker=dict(size=5),
            name=f'Group {group_index + 1}'
        ))
        particle_index += num_particles

    fig = go.Figure(
        data=initial_data,
        layout=go.Layout(
            updatemenus=[dict(
                type="buttons",
                showactive=False,
                buttons=[dict(label="Play",
                            method="animate",
                            args=[None, dict(frame=dict(duration=100, redraw=True), fromcurrent=True)])]
            )],
            scene=dict(
                xaxis=dict(range=[-2, 2]),
                yaxis=dict(range=[-2, 2]),
                zaxis=dict(range=[-1, 1]),
                aspectmode='cube' 
            ),
            title="3D Particle Trajectories"
        ),
        frames=frames
    )

    fig.show()





def reorganize_by_label(traj):
    label_dict = {}

    for particle in traj:
        for entry in particle:
            label = entry[-1]
            if label not in label_dict:
                label_dict[label] = []
            label_dict[label].append(entry)
    
    reorganized_result = []
    for label, values in label_dict.items():
        reorganized_result.append(values)

    return reorganized_result




import plotly.graph_objs as go

def plot_traj_labels_plotly_2nd(trajectories, duration=100, noise_label=-1, color_map = "Spectral"):
    num_timesteps = len(trajectories[0])
    num_particles = len(trajectories)
    groups_list = np.array(trajectories)[:, :, -1]
    unique_groups = np.unique(np.array(groups_list).flatten())
    num_groups = len(unique_groups)

    #colors = ['red', 'blue', 'green', 'yellow', 'orange', 'purple', 'brown', 'pink', 'grey', 'cyan']
    color_palette = plt.cm.get_cmap(color_map, num_groups)
    colors = [color_palette(i) for i in range(num_groups)]
    colors = [f'rgba({int(c[0]*255)}, {int(c[1]*255)}, {int(c[2]*255)}, {c[3]})' for c in colors]

    color_mapping = {group: 'black' if group == noise_label else colors[i] 
                     for i, group in enumerate(unique_groups)}

    xmin = np.min(np.array(trajectories)[:, :, 0].flatten())
    xmax = np.max(np.array(trajectories)[:, :, 0].flatten())
    ymin = np.min(np.array(trajectories)[:, :, 1].flatten())
    ymax = np.max(np.array(trajectories)[:, :, 1].flatten())
    zmin = np.min(np.array(trajectories)[:, :, 2].flatten())
    zmax = np.max(np.array(trajectories)[:, :, 2].flatten())

    frames = []
    for t in range(num_timesteps):
        result = [[particle[t]] for particle in trajectories]
        result = reorganize_by_label(result)

        num_particles_per_group = []
        for ii in result:
            num_particles_per_group.append(len(ii))

        frame_data = []
        for group_index in range(num_groups):
            group = unique_groups[group_index]
            particle_index = 0
            num_particles = num_particles_per_group[group_index]
            xs = [result[group_index][particle_index + p][0] for p in range(num_particles)]
            ys = [result[group_index][particle_index + p][1] for p in range(num_particles)]
            zs = [result[group_index][particle_index + p][2] for p in range(num_particles)]
            frame_data.append(go.Scatter3d(
                x=xs, y=ys, z=zs,
                mode='markers',
                marker=dict(size=5, color=color_mapping[group]),
                name=f'Group {group}'
            ))
            particle_index += num_particles
        frames.append(go.Frame(data=frame_data, name=str(t)))

    initial_data = []
    for group_index in range(num_groups):
        group = unique_groups[group_index]
        particle_index = 0
        result = [[particle[0]] for particle in trajectories]
        result = reorganize_by_label(result)
        num_particles_per_group = []
        for ii in result:
            num_particles_per_group.append(len(ii))

        num_particles = num_particles_per_group[group_index]
        xs = [result[group_index][particle_index + p][0] for p in range(num_particles)]
        ys = [result[group_index][particle_index + p][1] for p in range(num_particles)]
        zs = [result[group_index][particle_index + p][2] for p in range(num_particles)]
        initial_data.append(go.Scatter3d(
            x=xs, y=ys, z=zs,
            mode='markers',
            marker=dict(size=5, color=color_mapping[group]),
            name=f'Group {group}'
        ))
        particle_index += num_particles

    fig = go.Figure(
        data=initial_data,
        layout=go.Layout(
            updatemenus=[dict(
                type="buttons",
                showactive=False,
                buttons=[dict(label="Play",
                              method="animate",
                              args=[None, dict(frame=dict(duration=duration, redraw=True), fromcurrent=True)])]
            )],
            scene=dict(
                xaxis=dict(range=[xmin, xmax]),
                yaxis=dict(range=[ymin, ymax]),
                zaxis=dict(range=[zmin, zmax]),
                aspectmode='cube'
            ),
            title="3D Particle Trajectories"
        ),
        frames=frames
    )

    fig.show()




import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt

def plot_traj_labels_plotly_2nd(trajectories, duration=100, noise_label=-1, color_map="Spectral"):
    """
    Plots 2D or 3D particle trajectories using Plotly, automatically detecting dimensionality.
    
    Parameters:
        trajectories (list): List of trajectories where each trajectory contains points in [x, y, (z), label] format.
        duration (int): Animation frame duration in milliseconds.
        noise_label (int): Label used for noise points (colored black).
        color_map (str): Colormap for labeled groups.
    """
    num_timesteps = len(trajectories[0])
    num_particles = len(trajectories)
    
    # Detect 2D or 3D based on the number of coordinates per point
    sample_point = trajectories[0][0]
    is_3d = len(sample_point) == 4  # [x, y, z, label] → 3D, [x, y, label] → 2D

    # Extract labels and unique groups
    groups_list = np.array(trajectories)[:, :, -1]
    unique_groups = np.unique(groups_list.flatten())
    num_groups = len(unique_groups)

    # Generate colors for groups
    color_palette = plt.cm.get_cmap(color_map, num_groups)
    colors = [color_palette(i) for i in range(num_groups)]
    colors = [f'rgba({int(c[0]*255)}, {int(c[1]*255)}, {int(c[2]*255)}, {c[3]})' for c in colors]
    color_mapping = {group: 'black' if group == noise_label else colors[i] for i, group in enumerate(unique_groups)}

    # Get min/max values for axis limits
    xmin = np.min(np.array(trajectories)[:, :, 0])
    xmax = np.max(np.array(trajectories)[:, :, 0])
    ymin = np.min(np.array(trajectories)[:, :, 1])
    ymax = np.max(np.array(trajectories)[:, :, 1])
    
    if is_3d:
        zmin = np.min(np.array(trajectories)[:, :, 2])
        zmax = np.max(np.array(trajectories)[:, :, 2])

    frames = []
    for t in range(num_timesteps):
        result = [[particle[t]] for particle in trajectories]
        result = reorganize_by_label(result)  # Assumed external function

        num_particles_per_group = [len(group) for group in result]

        frame_data = []
        for group_index in range(num_groups):
            group = unique_groups[group_index]
            particle_index = 0
            num_particles = num_particles_per_group[group_index]

            xs = [result[group_index][particle_index + p][0] for p in range(num_particles)]
            ys = [result[group_index][particle_index + p][1] for p in range(num_particles)]
            zs = [result[group_index][particle_index + p][2] for p in range(num_particles)] if is_3d else None

            if is_3d:
                scatter = go.Scatter3d(
                    x=xs, y=ys, z=zs,
                    mode='markers',
                    marker=dict(size=5, color=color_mapping[group]),
                    name=f'Group {group}'
                )
            else:
                scatter = go.Scatter(
                    x=xs, y=ys,
                    mode='markers',
                    marker=dict(size=5, color=color_mapping[group]),
                    name=f'Group {group}'
                )
            frame_data.append(scatter)
            particle_index += num_particles
        frames.append(go.Frame(data=frame_data, name=str(t)))

    # Initial frame
    initial_data = []
    for group_index in range(num_groups):
        group = unique_groups[group_index]
        particle_index = 0
        result = [[particle[0]] for particle in trajectories]
        result = reorganize_by_label(result)
        num_particles_per_group = [len(group) for group in result]

        num_particles = num_particles_per_group[group_index]
        xs = [result[group_index][particle_index + p][0] for p in range(num_particles)]
        ys = [result[group_index][particle_index + p][1] for p in range(num_particles)]
        zs = [result[group_index][particle_index + p][2] for p in range(num_particles)] if is_3d else None

        if is_3d:
            scatter = go.Scatter3d(
                x=xs, y=ys, z=zs,
                mode='markers',
                marker=dict(size=5, color=color_mapping[group]),
                name=f'Group {group}'
            )
        else:
            scatter = go.Scatter(
                x=xs, y=ys,
                mode='markers',
                marker=dict(size=5, color=color_mapping[group]),
                name=f'Group {group}'
            )
        initial_data.append(scatter)
        particle_index += num_particles

    # Create figure
    layout = go.Layout(
        updatemenus=[dict(
            type="buttons",
            showactive=False,
            buttons=[dict(label="Play",
                          method="animate",
                          args=[None, dict(frame=dict(duration=duration, redraw=True), fromcurrent=True)])]
        )],
        title="Particle Trajectories"
    )

    if is_3d:
        layout.update(scene=dict(
            xaxis=dict(range=[xmin, xmax]),
            yaxis=dict(range=[ymin, ymax]),
            zaxis=dict(range=[zmin, zmax]),
            aspectmode='cube'
        ))
    else:
        layout.update(xaxis=dict(range=[xmin, xmax]), yaxis=dict(range=[ymin, ymax]))

    fig = go.Figure(data=initial_data, layout=layout, frames=frames)
    fig.show()





def plot_traj_labels_plt_eff(particles, interval=200, save_video=False, filename="example_plot.mp4", color_map='Spectral', noise_label=-1):
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm

    particles = np.array(particles)
    num_timesteps = particles.shape[1]
    num_particles = particles.shape[0]
    groups_list = particles[:, :, -1]
    num_groups = len(np.unique(groups_list.flatten()))
    min_group = int(np.min(groups_list.flatten()))
    dt = 0.1

    viridis = cm.get_cmap(color_map, num_groups)
    label_colors = viridis(np.linspace(0, 1, num_groups))

    x_min, x_max = particles[:, :, 0].min(), particles[:, :, 0].max()
    y_min, y_max = particles[:, :, 1].min(), particles[:, :, 1].max()
    z_min, z_max = particles[:, :, 2].min(), particles[:, :, 2].max()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)

    scatters = [ax.scatter([], [], [], marker='o') for _ in range(num_particles)]
    time_template = 'time = %.1f'
    time_text = ax.text(0.05, 0.9, 0.9, '', transform=ax.transAxes)

    particle_positions = [particles[:, i, :3] for i in range(num_timesteps)]
    particle_labels = [particles[:, i, -1] for i in range(num_timesteps)]

    def init():
        time_text.set_text('')
        return scatters + [time_text]

    def _update_plot(i):
        for j in range(num_particles):
            pos = particle_positions[i][j]
            label = int(particle_labels[i][j])
            scatters[j]._offsets3d = ([pos[0]], [pos[1]], [pos[2]])
            color = 'black' if noise_label != -1 and label == noise_label else label_colors[label - min_group]
            scatters[j].set_color(color)
        
        time_text.set_text(time_template % (i * dt))
        return scatters + [time_text]

    anim = animation.FuncAnimation(fig, _update_plot, frames=num_timesteps, interval=interval, blit=True, init_func=init)
    plt.show()

    if save_video:
        anim.save(filename)

    return anim



def format_title(title, max_length=50):
    words = title.split()
    lines = []
    current_line = []

    for word in words:
        if sum(len(w) for w in current_line) + len(word) + len(current_line) > max_length:
            lines.append(' '.join(current_line))
            current_line = [word]
        else:
            current_line.append(word)
    
    if current_line:
        lines.append(' '.join(current_line))
    
    return '\n'.join(lines)





def plot_traj_labels_plt(particles, interval = 200, save_video=False, filename="example_plot.mp4", color_map='Spectral', noise_label = -1, title = None, apply_format_title=True, plt_fig_size=(8, 6)):
    import numpy as np 
    num_timesteps = len(particles[0])
    num_particles = len(particles)
    groups_list = np.array(particles)[:,:,-1]
    num_groups = len(np.unique(np.array(groups_list).flatten()))

    min_group = int(np.min(groups_list.flatten()))


    from matplotlib import cm
    from matplotlib.colors import ListedColormap, LinearSegmentedColormap


    viridis = cm.get_cmap(color_map, len(particles))
    #label_colors = viridis(np.linspace(0, 1, len(particles)))
    label_colors = viridis(np.linspace(0, 1, num_groups))


    dt = 0.1

    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    import numpy as np
    from mpl_toolkits.mplot3d import Axes3D


    def init():
        time_text.set_text('')
        return time_text


    particle_lists = [[] for _ in range(len(particles))] 
    label_lists = [[] for _ in range(len(particles))]  

    #fig = plt.figure()
    fig = plt.figure(figsize=plt_fig_size)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(-25, 25)
    ax.set_ylim(-25, 25)
    ax.set_zlim(-25, 25)

    ax.set_xlim(np.min(np.array(particles)[:,:,0].flatten()), np.max(np.array(particles)[:,:,0].flatten()))
    ax.set_ylim(np.min(np.array(particles)[:,:,1].flatten()), np.max(np.array(particles)[:,:,1].flatten()))
    ax.set_zlim(np.min(np.array(particles)[:,:,2].flatten()), np.max(np.array(particles)[:,:,2].flatten()))

    if title:
        if apply_format_title:
            title_lines = title.split("\n")
            if len(title_lines)  == 2:
                title = format_title(title_lines[0], max_length=80) + "\n" + title_lines[1]
            else: 
                title = format_title(title_lines[0], max_length=80)
        ax.set_title(title, fontsize=10)


    scatters = []  
    for _ in range(len(particles)):
        scatters.append(ax.scatter([], [], [], marker='o'))

    time_template = 'time = %.1f'
    time_text = ax.text(0.05, 0.9, 0.9, '', transform=ax.transAxes)

    def _update_plot(i):
        
        for j, particle in enumerate(particles):
            
            particle_list = particle_lists[j]
            
            label_list = label_lists[j]
            
            particle_list.clear()  
            label_list.clear() 
            
            newp = []
            newl = []
            for entry in particle:
                newp.append(entry[:3])
                newl.append(int(entry[-1]))
                
            particle_list.append(newp[i])     
            label_list.append(newl[i])
            #particle_list.append(entry[:3]) 
            
            #particle_list.append(particle[i])
            
            #particle_list = particle_lists[j]
            #particle_list.append(particle[i])
            scatters[j]._offsets3d = tuple(zip(*particle_list))
            #scatters[j].set_array(np.array(label_list))
            #scatters[j].set_color([0,1,0,0.5])
            if noise_label:
                if newl[i] == noise_label:
                    scatters[j].set_color('black')
                else:
                    scatters[j].set_color(label_colors[newl[i]-min_group])
            else:
                scatters[j].set_color(label_colors[newl[i]-min_group])
        time_text.set_text(time_template % (i * dt))


    anim = animation.FuncAnimation(fig, _update_plot,
                               frames=len(particles[0]), interval=interval, blit=True, init_func=init)

    plt.show()


    from IPython.display import HTML
    #HTML(anim.to_html5_video())

    if save_video:
        anim.save(filename)

    return anim


import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np


def ani_2d_plot(df, curr_file, filename_add="", save_plot=True, title_add="animated_plot_cakmak.mp4"):
    unique_labels = df['label'].unique()
    colors = plt.cm.tab10(range(len(unique_labels)))
    label_color_map = dict(zip(unique_labels, colors))

    fig, ax = plt.subplots(figsize=(8, 6))

    # Function to update the scatter plot
    def update(frame):
        ax.clear()
        current_data = df[df['t'] == frame]
        scatter = ax.scatter(
            current_data['x'],
            current_data['y'],
            c=[label_color_map[label] for label in current_data['label']],
            s=100,  # Adjust size as needed
            edgecolor='k'
        )
        ax.set_title(f"Timeframe: {frame}")
        ax.set_xlim(df['x'].min() - 1, df['x'].max() + 1)
        ax.set_ylim(df['y'].min() - 1, df['y'].max() + 1)
        ax.set_xlabel("X-coordinate")
        ax.set_ylabel("Y-coordinate")
        return scatter,

    frames = df['t'].unique()
    ani = FuncAnimation(fig, update, frames=frames, repeat=True, interval=500)

    # Save or display the animation
    if save_plot:
        ani.save(filename_add + curr_file.split('.')[0] + title_add, writer='ffmpeg')

    plt.show()