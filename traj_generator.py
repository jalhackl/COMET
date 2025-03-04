import numpy as np
import plotly.graph_objs as go



'''
def linear_motion(t, length=2, speed=1, reverse=False, y_speed = 0, z_speed=0):
    period = 2 * length / speed
    t = t % period
    if reverse:
        t = period - t
    if t < period / 2:
        x = speed * t - length / 2
    else:
        x = length / 2 - speed * (t - period / 2)
    y = y_speed * t
    z = z_speed * t
    return x, y, z



def linear_motion(t, length=2, speed=1, reverse=False, z_speed=0, y_speed=0):
    period = 2 * length / speed
    t = t % period
    direction = 1 if t < period / 2 else -1
    if reverse:
        direction *= -1
    t = t if t < period / 2 else period - t
    x = direction * (speed * t - length / 2)
    y = direction * y_speed * t
    z = direction * z_speed * t
    return x, y, z
'''

def circular_motion(t, radius=1, speed=1, clockwise=True, z_speed=0, perturbation=0):
    angle = speed * t * (1 if clockwise else -1)
    x = radius * np.cos(angle) + np.random.uniform(-perturbation, perturbation)
    y = radius * np.sin(angle) + np.random.uniform(-perturbation, perturbation)
    z = z_speed * t + np.random.uniform(-perturbation, perturbation)
    return x, y, z


def circular_motion(
    t, 
    radius=1, 
    speed=1, 
    clockwise=True, 
    z_speed=0, 
    perturbation=0, 
    initial_pos=(0, 0, 0), 
    initial_velocity=(0, 0, 0)
):
    angle = speed * t * (1 if clockwise else -1)
    x = radius * np.cos(angle) + np.random.uniform(-perturbation, perturbation)
    y = radius * np.sin(angle) + np.random.uniform(-perturbation, perturbation)
    z = z_speed * t + np.random.uniform(-perturbation, perturbation)
    
    # Add initial conditions
    x += initial_pos[0] + initial_velocity[0] * t
    y += initial_pos[1] + initial_velocity[1] * t
    z += initial_pos[2] + initial_velocity[2] * t
    
    return x, y, z




def linear_motion(t, length=2, y_length=2, z_length=2, speed=1, y_speed=1, z_speed=1, reverse=False, perturbation=0):
    x_length = length
    x_speed = speed


    x_period = 2 * x_length / x_speed
    y_period = 2 * y_length / y_speed
    z_period = 2 * z_length / z_speed
    
    # Calculate x position
    x_t = t % x_period
    if x_t < x_period / 2:
        x = x_speed * x_t - x_length / 2
    else:
        x = x_length / 2 - x_speed * (x_t - x_period / 2)
    if reverse:
        x = -x
    
    # Calculate y position
    y_t = t % y_period
    if y_t < y_period / 2:
        y = y_speed * y_t - y_length / 2
    else:
        y = y_length / 2 - y_speed * (y_t - y_period / 2)
    
    # Calculate z position
    z_t = t % z_period
    if z_t < z_period / 2:
        z = z_speed * z_t - z_length / 2
    else:
        z = z_length / 2 - z_speed * (z_t - z_period / 2)
    
    # perturbations
    x += np.random.uniform(-perturbation, perturbation)
    y += np.random.uniform(-perturbation, perturbation)
    z += np.random.uniform(-perturbation, perturbation)
    
    return x, y, z



def linear_motion(
    t, 
    length=2, 
    y_length=2, 
    z_length=2, 
    speed=1, 
    y_speed=1, 
    z_speed=1, 
    reverse=False, 
    perturbation=0, 
    initial_pos=(0, 0, 0), 
    initial_velocity=(0, 0, 0)
):
    x_length = length
    x_speed = speed
    
    x_period = 2 * x_length / x_speed
    y_period = 2 * y_length / y_speed
    z_period = 2 * z_length / z_speed
    
    # Calculate x position
    x_t = t % x_period
    if x_t < x_period / 2:
        x = x_speed * x_t - x_length / 2
    else:
        x = x_length / 2 - x_speed * (x_t - x_period / 2)
    if reverse:
        x = -x
    
    # Calculate y position
    y_t = t % y_period
    if y_t < y_period / 2:
        y = y_speed * y_t - y_length / 2
    else:
        y = y_length / 2 - y_speed * (y_t - y_period / 2)
    
    # Calculate z position
    z_t = t % z_period
    if z_t < z_period / 2:
        z = z_speed * z_t - z_length / 2
    else:
        z = z_length / 2 - z_speed * (z_t - z_period / 2)
    
    # perturbations
    x += np.random.uniform(-perturbation, perturbation)
    y += np.random.uniform(-perturbation, perturbation)
    z += np.random.uniform(-perturbation, perturbation)
    
    # Add initial conditions
    x += initial_pos[0] + initial_velocity[0] * t
    y += initial_pos[1] + initial_velocity[1] * t
    z += initial_pos[2] + initial_velocity[2] * t
    
    return x, y, z



def generate_trajectories(num_particles, num_timesteps, motion_func, group_label, **kwargs):
    trajectories = []
    for p in range(num_particles):
        particle_trajectory = []
        initial_x, initial_y, initial_z = np.random.uniform(-0.5, 0.5, 3)
        for t in range(num_timesteps):
            x, y, z = motion_func(t, **kwargs)
            x += initial_x
            y += initial_y
            z += initial_z + np.random.uniform(-0.05, 0.05)
            particle_trajectory.append([x, y, z, group_label])
        trajectories.append(particle_trajectory)
    return trajectories

def generate_all_trajectories(num_particles_per_group, num_timesteps):
    all_trajectories = []
    group_label = 1
    
    for num_particles, motion_func, kwargs in num_particles_per_group:
        group_trajectories = generate_trajectories(num_particles, num_timesteps, motion_func, group_label, **kwargs)
        all_trajectories.extend(group_trajectories)
        group_label += 1
    
    return all_trajectories




def generate_all_trajectories_with_switch_add_points(
    num_particles_per_group_phase1, 
    num_particles_per_group_phase2, 
    num_timesteps_per_phase
):
    all_trajectories = []
    group_label = 1
    
    # Generate trajectories for the first n timesteps
    for num_particles, motion_func, kwargs in num_particles_per_group_phase1:
        group_trajectories = generate_trajectories(
            num_particles, num_timesteps_per_phase, motion_func, group_label, **kwargs
        )
        all_trajectories.extend(group_trajectories)
        group_label += 1
    
    # Generate trajectories for the next n timesteps
    # Start with the last positions of Phase 1 trajectories
    group_label = 1
    for num_particles, motion_func, kwargs in num_particles_per_group_phase2:
        phase1_group_trajectories = [
            traj[-1][:3]  # Extract the last x, y, z position for each particle
            for traj in all_trajectories 
            if traj[0][3] == group_label  # Match group_label
        ]
        
        group_trajectories = []
        for p, initial_pos in enumerate(phase1_group_trajectories):
            particle_trajectory = []
            for t in range(num_timesteps_per_phase):
                x, y, z = motion_func(t, **kwargs)
                x += initial_pos[0]
                y += initial_pos[1]
                z += initial_pos[2] + np.random.uniform(-0.05, 0.05)
                particle_trajectory.append([x, y, z, group_label])
            group_trajectories.append(particle_trajectory)
        
        all_trajectories.extend(group_trajectories)
        group_label += 1

    return all_trajectories

def generate_all_trajectories_with_switch(
    num_particles_per_group_phase1, 
    num_particles_per_group_phase2, 
    num_timesteps_per_phase
):
    """
    Generates trajectories for two phases, with movements switched after the first phase.

    :param num_particles_per_group_phase1: List of tuples (num_particles, motion_func, kwargs) for phase 1.
    :param num_particles_per_group_phase2: List of tuples (num_particles, motion_func, kwargs) for phase 2.
    :param num_timesteps_per_phase: Number of timesteps per phase.
    :return: A combined trajectory list for all particles across both phases.
    """
    all_trajectories = []
    particle_indices = []
    current_idx = 0

    # Generate trajectories for the first phase
    for num_particles, motion_func, kwargs in num_particles_per_group_phase1:
        group_trajectories = generate_trajectories(
            num_particles, num_timesteps_per_phase, motion_func, current_idx, **kwargs
        )
        all_trajectories.extend(group_trajectories)
        particle_indices.extend([current_idx] * num_particles)
        current_idx += 1

    # Prepare to continue trajectories in phase 2
    phase1_final_positions = [
        traj[-1][:3] for traj in all_trajectories
    ]  # Extract the last x, y, z positions of Phase 1

    # Update trajectories with switched motions
    num_particles_list_phase1 = [group[0] for group in num_particles_per_group_phase1]  # Extract particle counts
    for (num_particles, motion_func, kwargs), group_idx in zip(
        num_particles_per_group_phase2, range(len(num_particles_per_group_phase2))
    ):
        start_idx = sum(num_particles_list_phase1[:group_idx])  # Use extracted particle counts
        end_idx = start_idx + num_particles

        for i in range(start_idx, end_idx):
            initial_pos = phase1_final_positions[i]
            for t in range(num_timesteps_per_phase):
                x, y, z = motion_func(t, **kwargs)
                x += initial_pos[0]
                y += initial_pos[1]
                z += initial_pos[2] + np.random.uniform(-0.05, 0.05)
                all_trajectories[i].append([x, y, z, particle_indices[i]])

    return all_trajectories




def generate_all_trajectories_with_permutation(
    num_particles_per_group, 
    num_timesteps_per_phase,
    motion_funcs_phase1,
    motion_funcs_phase2
):
    """
    Generates particle trajectories for two phases, with group membership permuted
    after the first phase.
    """
    total_particles = sum(num_particles_per_group)
    all_trajectories = [[] for _ in range(total_particles)]
    group_assignments = []
    
    # Generate initial trajectories
    group_start_idx = 0
    phase1_group_assignments = []
    for group_idx, (num_particles, (motion_func, kwargs)) in enumerate(
        zip(num_particles_per_group, motion_funcs_phase1), 1
    ):
        group_end_idx = group_start_idx + num_particles
        phase1_group_assignments.extend([group_idx] * num_particles)
        
        group_trajectories = generate_trajectories(
            num_particles, num_timesteps_per_phase, motion_func, group_idx, **kwargs
        )
        for i, traj in enumerate(group_trajectories):
            all_trajectories[group_start_idx + i] = traj
        group_start_idx = group_end_idx

    group_assignments.append(phase1_group_assignments)
    
    # Permute group assignments for Phase 2
    permuted_indices = np.random.permutation(total_particles)
    phase2_group_assignments = [None] * total_particles
    
    group_start_idx = 0
    for group_idx, num_particles in enumerate(num_particles_per_group, 1):
        group_end_idx = group_start_idx + num_particles
        for i in permuted_indices[group_start_idx:group_end_idx]:
            phase2_group_assignments[i] = group_idx
        group_start_idx = group_end_idx

    group_assignments.append(phase2_group_assignments)

    # Generate trajectories for permuted groups
    phase1_final_positions = [
        traj[-1][:3] for traj in all_trajectories
    ]
    phase1_final_velocities = [
        (traj[-1][0] - traj[-2][0], traj[-1][1] - traj[-2][1], traj[-1][2] - traj[-2][2])
        for traj in all_trajectories
    ]

    for i, (motion_func, kwargs) in enumerate(motion_funcs_phase2):
        group_particles = [
            j for j, g in enumerate(phase2_group_assignments) if g == i + 1
        ]
        for p in group_particles:
            initial_pos = phase1_final_positions[p]
            initial_velocity = phase1_final_velocities[p]
            particle_trajectory = []
            for t in range(num_timesteps_per_phase):
                x, y, z = motion_func(
                    t,
                    initial_pos=initial_pos,
                    initial_velocity=initial_velocity,
                    **kwargs
                )
                particle_trajectory.append([x, y, z, i + 1])
            all_trajectories[p].extend(particle_trajectory)

    return all_trajectories, group_assignments



def generate_trajectories_variation(num_particles, num_timesteps, motion_funcs, group_label, kwargs_lists):
    """
    Generate trajectories for a group of particles, allowing switching between motion functions or parameters.

    Parameters:
    - num_particles: Number of particles in the group.
    - num_timesteps: Total number of timesteps for the simulation.
    - motion_funcs: List of tuples (motion_func, duration), where motion_func is a function and
                     duration is the number of timesteps to apply that function.
    - group_label: Label for the group of particles.
    - kwargs_lists: List of lists of dictionaries, each containing kwargs for the corresponding motion function.

    Returns:
    - List of trajectories for all particles in the group.
    """
    trajectories = []
    for p in range(num_particles):
        particle_trajectory = []
        initial_x, initial_y, initial_z = np.random.uniform(-0.5, 0.5, 3)
        t_offset = 0  # Track the start time for each motion segment

        # Iterate through the motion functions and their corresponding kwargs
        for (motion_func, duration), kwargs in zip(motion_funcs, kwargs_lists):
            for t in range(duration):
                if t_offset + t >= num_timesteps:
                    break  # Stop if we exceed the total timesteps
                # Ensure we correctly pass the kwargs dictionary to the motion function
                x, y, z = motion_func(t + t_offset, **kwargs)
                x += initial_x
                y += initial_y
                z += initial_z + np.random.uniform(-0.05, 0.05)
                particle_trajectory.append([x, y, z, group_label])
            t_offset += duration
            if t_offset >= num_timesteps:
                break  # Stop if we've generated enough timesteps

        # Ensure trajectory has the required number of timesteps
        if len(particle_trajectory) < num_timesteps:
            while len(particle_trajectory) < num_timesteps:
                particle_trajectory.append(particle_trajectory[-1])

        trajectories.append(particle_trajectory)
    return trajectories


def generate_all_trajectories_variation(num_particles_per_group, num_timesteps):
    """
    Generate trajectories for all groups of particles.

    Parameters:
    - num_particles_per_group: List of tuples (num_particles, motion_funcs, kwargs_lists),
                               where motion_funcs and kwargs_lists define the motion behaviors for a group.
    - num_timesteps: Total number of timesteps for the simulation.

    Returns:
    - List of trajectories for all particles.
    """
    all_trajectories = []
    group_label = 1

    for num_particles, motion_funcs, kwargs_lists in num_particles_per_group:
        group_trajectories = generate_trajectories_variation(
            num_particles, num_timesteps, motion_funcs, group_label, kwargs_lists
        )
        all_trajectories.extend(group_trajectories)
        group_label += 1

    return all_trajectories