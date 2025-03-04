import numpy as np


def circular_motion(
    t,
    radius=1,
    speed=1,
    clockwise=True,
    z_speed=0,
    perturbation=0,
    initial_pos=(0, 0, 0),
    initial_dir=(1, 0, 0),
):
    """
    Generates circular motion in a local frame, transformed to the global frame.
    """
    angle = speed * t * (1 if clockwise else -1)
    local_x = radius * np.cos(angle) + np.random.uniform(-perturbation, perturbation)
    local_y = radius * np.sin(angle) + np.random.uniform(-perturbation, perturbation)
    local_z = z_speed * t + np.random.uniform(-perturbation, perturbation)

    # Define the rotation matrix to map local coordinates to global frame
    direction = np.array(initial_dir) / np.linalg.norm(initial_dir)
    z_axis = np.array([0, 0, 1])
    if np.allclose(direction, z_axis):
        x_axis = np.array([1, 0, 0])
    else:
        x_axis = np.cross(z_axis, direction)
        x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(direction, x_axis)

    rotation_matrix = np.stack([x_axis, y_axis, direction], axis=1)

    local_coords = np.array([local_x, local_y, local_z])
    global_coords = initial_pos + rotation_matrix @ local_coords

    return global_coords.tolist()


def linear_motion(
    t,
    length=2,
    speed=1,
    y_speed=0,
    z_speed=0,
    reverse=False,
    perturbation=0,
    initial_pos=(0, 0, 0),
    initial_dir=(1, 0, 0),
):
    """
    Generates linear motion in a local frame, transformed to the global frame.
    """
    local_x = speed * t if not reverse else -speed * t
    local_y = y_speed * t
    local_z = z_speed * t

    # Apply periodic boundary conditions for length
    if length > 0:
        local_x = (local_x + length / 2) % length - length / 2

    # Add perturbation
    local_x += np.random.uniform(-perturbation, perturbation)
    local_y += np.random.uniform(-perturbation, perturbation)
    local_z += np.random.uniform(-perturbation, perturbation)

    # Define the rotation matrix to map local coordinates to global frame
    direction = np.array(initial_dir) / np.linalg.norm(initial_dir)
    z_axis = np.array([0, 0, 1])
    if np.allclose(direction, z_axis):
        x_axis = np.array([1, 0, 0])
    else:
        x_axis = np.cross(z_axis, direction)
        x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(direction, x_axis)

    rotation_matrix = np.stack([x_axis, y_axis, direction], axis=1)

    local_coords = np.array([local_x, local_y, local_z])
    global_coords = initial_pos + rotation_matrix @ local_coords

    return global_coords.tolist()


def generate_all_trajectories_with_permutation_local(
    num_particles_per_group,
    num_timesteps_per_phase,
    motion_funcs_phase1,
    motion_funcs_phase2,
    group_spread=0.1  # Optional parameter to control how spread out particles are within each group
):
    """
    Generates particle trajectories for two phases with smooth transitions between phases,
    ensuring particles within a group are initialized with unique positions.
    """
    total_particles = sum(num_particles_per_group)
    all_trajectories = []
    group_assignments = []

    # Phase 1: Generate initial trajectories
    group_start_idx = 0
    phase1_group_assignments = []
    for group_idx, (num_particles, (motion_func, kwargs)) in enumerate(
        zip(num_particles_per_group, motion_funcs_phase1), 1
    ):
        group_end_idx = group_start_idx + num_particles
        phase1_group_assignments.extend([group_idx] * num_particles)

        # Initialize positions for the particles in the group with slight random offsets
        group_trajectories = []
        for _ in range(num_particles):
            # Add a small random offset to the initial positions within the group
            offset = np.random.uniform(-group_spread, group_spread, 3)
            initial_pos = offset  # Adjust this if you want a different central position
            particle_trajectory = [
                motion_func(t, initial_pos=initial_pos, **kwargs) + [group_idx]
                for t in range(num_timesteps_per_phase)
            ]
            group_trajectories.append(particle_trajectory)

        all_trajectories.extend(group_trajectories)
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

    # Phase 2: Generate trajectories for permuted groups
    phase1_final_positions = [
        traj[-1][:3] for traj in all_trajectories
    ]  # Extract last x, y, z positions from Phase 1
    phase1_final_directions = [
        [
            traj[-1][dim] - traj[-2][dim]
            for dim in range(3)
        ]
        for traj in all_trajectories
    ]

    for i, (motion_func, kwargs) in enumerate(motion_funcs_phase2):
        group_particles = [
            j for j, g in enumerate(phase2_group_assignments) if g == i + 1
        ]
        for p in group_particles:
            initial_pos = phase1_final_positions[p]
            initial_dir = phase1_final_directions[p]
            particle_trajectory = []
            for t in range(num_timesteps_per_phase):
                position = motion_func(
                    t,
                    initial_pos=initial_pos,
                    initial_dir=initial_dir,
                    **kwargs,
                )
                particle_trajectory.append(position + [i + 1])
            all_trajectories[p].extend(particle_trajectory)

    return all_trajectories, group_assignments

