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