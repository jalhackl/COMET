import numpy as np
import pandas as pd


def mix_timesteps(data):

    data = np.array(data)

    num_timesteps = data.shape[1]

    permutation = np.random.permutation(num_timesteps)

    mixed_data = data[:, permutation, :]
    
    return mixed_data


def thin_timesteps_randomly(data, num_timesteps):

    data = np.array(data)
    

    total_timesteps = data.shape[1]

    selected_timesteps = np.random.choice(total_timesteps, num_timesteps, replace=False)
    

    thinned_data = data[:, selected_timesteps, :]
    
    return thinned_data




def create_clustering_df(trajectories):
    flattened_data = []

    for obj_id, particle_data in enumerate(trajectories):
        for t, record in enumerate(particle_data):
            # Ensure there are at least 3 coordinates (x, y) and a label
            if len(record) == 3:
                # If z is missing, set it to 0
                particle_data[t] = [record[0], record[1], 0, record[2]]
            # Append the flattened data
            flattened_data.append([obj_id, t] + particle_data[t])

    df = pd.DataFrame(flattened_data, columns=['obj_id', 't', 'x', 'y', 'z', 'label'])

    return df


def create_clustering_df_ndim(trajectories):
    """
    Converts a list of trajectories into a flat pandas DataFrame with automatic dimensionality detection.
    
    Parameters:
        trajectories (list): Each trajectory is a list of records [x1, x2, ..., xn, label]

    Returns:
        pd.DataFrame: Columns = ['obj_id', 't', 'x1', ..., 'xn', 'label']
    """
    flattened_data = []

    # Determine dimensionality from the first record
    first_record = trajectories[0][0]
    n_dim = len(first_record) - 1  # Last entry is assumed to be the label

    for obj_id, particle_data in enumerate(trajectories):
        for t, record in enumerate(particle_data):
            coords = record[:-1]
            label = record[-1]

            # Pad with zeros if needed
            if len(coords) < n_dim:
                coords += [0] * (n_dim - len(coords))
            elif len(coords) > n_dim:
                coords = coords[:n_dim]

            flattened_data.append([obj_id, t] + coords + [label])

    # Create column names like ['x1', ..., 'xn']
    columns = ['obj_id', 't'] + [f'x{i+1}' for i in range(n_dim)] + ['label']
    return pd.DataFrame(flattened_data, columns=columns)


def create_clustering_df_simple(trajectories):
    flattened_data = []
    trajectories = trajectories.tolist()
    for obj_id, particle_data in enumerate(trajectories):
        for t, record in enumerate(particle_data):
            
            flattened_data.append([obj_id, t] + record)

    df = pd.DataFrame(flattened_data, columns=['obj_id', 't', 'x', 'y', 'z', 'label'])

    return df



def prepare_clustered_results_for_plotly(trajectories, result_labels):
    trajectories_results = [
    [[*timestep[:-1], result_labels[ip]] for timestep in particle_traj]
    for ip, particle_traj in enumerate(trajectories)
    ]

    #equivalent
    '''
    trajectories_results = []
    for ip, particle_traj in enumerate(trajectories):
        trajectory_result = []

    for it, timestep in enumerate(particle_traj):
        trajectory_result.append([*timestep[:-1], result_labels[ip]])
    trajectories_results.append(trajectory_result)
    '''
    return trajectories_results




def simulation_to_array(particles_from_sim, group_attribute_name = "move_group"):
    particle_list = list(particles_from_sim)

    positions_list = []
    times_list = []
    groups_list = []
    for pp in particle_list:
        position_list = []
        time_list = []
        group_list = []
        for timess, position in enumerate(pp.positions):
            position_list.append(position)
            time_list.append(timess)

            group_list.append(getattr(pp, group_attribute_name))

        positions_list.append(position_list)
        times_list.append(time_list)
        groups_list.append(group_list)

    particles_new = positions_list
    labels_new = list(range(len(particles_new)))

    particle_labels = []
    for ig, group in enumerate(particles_new):
        particle_label_group = []
        for ie, entry in enumerate(group):
            #new_entry = [*entry, labels_new[ig]]
            new_entry = [*entry, groups_list[ig][ie]]
            
            particle_label_group.append(new_entry)
        particle_labels.append(particle_label_group)

    trajectories = particle_labels

    return trajectories



def simulation_to_array_group_changes(particles_from_sim, group_attribute_name="move_group"):
    particle_list = list(particles_from_sim)

    positions_list = []
    times_list = []
    groups_list = []
    
    for pp in particle_list:
        position_list = []
        time_list = []
        group_list = []
        
        group_values = getattr(pp, group_attribute_name)  # Now a list

        for timess, (position, group) in enumerate(zip(pp.positions, group_values)):
            position_list.append(position)
            time_list.append(timess)
            group_list.append(group)  # Append corresponding group entry at this time step

        positions_list.append(position_list)
        times_list.append(time_list)
        groups_list.append(group_list)

    particles_new = positions_list
    labels_new = list(range(len(particles_new)))

    particle_labels = []
    for ig, group in enumerate(particles_new):
        particle_label_group = []
        for ie, entry in enumerate(group):
            new_entry = [*entry, groups_list[ig][ie]]  # Append corresponding group entry
            
            particle_label_group.append(new_entry)
        particle_labels.append(particle_label_group)

    trajectories = particle_labels

    return trajectories




def simulation_to_array_velocities(particles_from_sim, group_attribute_name = "move_group"):
    particle_list = list(particles_from_sim)

    velocities_list = []
    times_list = []
    groups_list = []
    for pp in particle_list:
        velocity_list = []
        time_list = []
        group_list = []
        for timess, position in enumerate(pp.velocities):
            velocity_list.append(position)
            time_list.append(timess)

            group_list.append(getattr(pp, group_attribute_name))

        velocities_list.append(velocity_list)
        times_list.append(time_list)
        groups_list.append(group_list)

    particles_new = velocities_list
    labels_new = list(range(len(particles_new)))

    particle_labels = []
    for ig, group in enumerate(particles_new):
        particle_label_group = []
        for ie, entry in enumerate(group):
            #new_entry = [*entry, labels_new[ig]]
            new_entry = [*entry, groups_list[ig][ie]]
            
            particle_label_group.append(new_entry)
        particle_labels.append(particle_label_group)

    trajectories = particle_labels

    return trajectories


def simulation_to_array_velocities_extended(particles_from_sim, include_positions=False, group_attribute_name="move_group"):
    particle_list = list(particles_from_sim)

    velocities_list = []
    positions_list = []
    times_list = []
    groups_list = []

    for pp in particle_list:
        velocity_list = []
        position_list = []
        time_list = []
        group_list = []

        has_dynamic_groups = hasattr(pp, "curr_group_list")  # Check if dynamic groups exist
        static_group = getattr(pp, group_attribute_name, None)  # Fallback if dynamic groups are absent

        for timess, velocity in enumerate(pp.velocities):
            velocity_list.append(velocity)
            time_list.append(timess)

            if has_dynamic_groups:
                group_list.append(pp.curr_group_list[timess])  # Use dynamic group
            else:
                group_list.append(static_group)  # Use static group

            if include_positions:
                position_list.append(pp.positions[timess])  # Extract position at this timestep

        velocities_list.append(velocity_list)
        times_list.append(time_list)
        groups_list.append(group_list)
        if include_positions:
            positions_list.append(position_list)

    particles_new = velocities_list
    labels_new = list(range(len(particles_new)))

    particle_labels = []
    for ig, group in enumerate(particles_new):
        particle_label_group = []
        for ie, entry in enumerate(group):
            new_entry = [*entry]  # Start with velocity data
            if include_positions:
                new_entry.extend(positions_list[ig][ie])  # Append position data
            new_entry.append(groups_list[ig][ie])  # Append group data
            
            particle_label_group.append(new_entry)
        particle_labels.append(particle_label_group)

    trajectories = particle_labels

    return trajectories



def compute_velocities(points, delta_t=1):

    velocities = []
    
    for point_series in points:
        point_velocities = []
        for i in range(1, len(point_series)):
            velocity = [(point_series[i][j] - point_series[i-1][j]) / delta_t for j in range(3)]
            point_velocities.append(velocity)
        
        #let's assume starting velocity is 0
        point_velocities.insert(0, [0.0, 0.0, 0.0])
        velocities.append(point_velocities)
    
    return velocities


def thin_trajectory(trajectory, thinning_factor):
  trajectory = trajectory[:, 0:trajectory.shape[1]:int(trajectory.shape[1] / thinning_factor), :]
  return trajectory