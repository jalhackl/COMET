import numpy as np


def initialize_simulate(num_points = 100, num_groups = 4, space_size = 10, timesteps = 200, noise_level = 0.1, group_change_interval = 50 ):
    # Initialize points
    positions = np.random.rand(num_points, 3) * space_size  # Random 3D positions
    velocities = np.random.rand(num_points, 3) - 0.5        # Random initial velocities
    group_assignments = np.random.randint(0, num_groups, num_points)  # Random group assignments

    # Shared parameters for group motions
    group_motion_types = np.random.choice([1, 2, 3, 4], size=num_groups)  # Assign strategies (1, 2, 3, or 4)
    group_base_velocities = np.random.rand(num_groups, 3) - 0.5  # Shared velocities for Strategy 1
    group_directions = np.random.rand(num_groups, 3) - 0.5       # Shared directions for Strategy 3
    group_directions = group_directions / np.linalg.norm(group_directions, axis=1)[:, np.newaxis]  # Normalize directions

    # Circular motion parameters
    circular_directions = [np.random.rand(3) for _ in range(num_groups)]  # Rotation axes for Strategy 4
    circular_directions = [d / np.linalg.norm(d) for d in circular_directions]  # Normalize rotation axes

    # Initialize storage for the simulation
    trajectory_data = np.zeros((num_points, timesteps, 4))  # (x, y, z, cluster)

    def rotation_matrix_around_axis(axis, angle):
        """Create a 3D rotation matrix around a given axis."""
        axis = axis / np.linalg.norm(axis)
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        u = axis
        return np.array([
            [cos_a + u[0]**2 * (1 - cos_a), u[0] * u[1] * (1 - cos_a) - u[2] * sin_a, u[0] * u[2] * (1 - cos_a) + u[1] * sin_a],
            [u[1] * u[0] * (1 - cos_a) + u[2] * sin_a, cos_a + u[1]**2 * (1 - cos_a), u[1] * u[2] * (1 - cos_a) - u[0] * sin_a],
            [u[2] * u[0] * (1 - cos_a) - u[1] * sin_a, u[2] * u[1] * (1 - cos_a) + u[0] * sin_a, cos_a + u[2]**2 * (1 - cos_a)],
        ])

    def update_positions(positions, velocities, group_assignments, noise_level):
        """Update the positions of the points."""
        group_centers = np.zeros((num_groups, 3))
        group_counts = np.zeros(num_groups)
        
        # Compute group centers (for Strategy 2: Group Center Attraction)
        for i, pos in enumerate(positions):
            group = group_assignments[i]
            group_centers[group] += pos
            group_counts[group] += 1
        
        for g in range(num_groups):
            if group_counts[g] > 0:
                group_centers[g] /= group_counts[g]  # Average position for each group

        # Update positions based on group strategy
        for i, pos in enumerate(positions):
            group = group_assignments[i]
            strategy = group_motion_types[group]
            
            if strategy == 1:  # Shared Group Velocity Vector
                velocities[i] = group_base_velocities[group] + noise_level * (np.random.rand(3) - 0.5)
            
            elif strategy == 2:  # Group Center Attraction
                direction_to_center = group_centers[group] - pos
                if np.linalg.norm(direction_to_center) > 1e-6:  # Avoid division by zero
                    velocities[i] = direction_to_center / np.linalg.norm(direction_to_center) + noise_level * (np.random.rand(3) - 0.5)
            
            elif strategy == 3:  # Shared Direction with Noise
                velocities[i] = group_directions[group] + noise_level * (np.random.rand(3) - 0.5)
                velocities[i] /= np.linalg.norm(velocities[i])  # Normalize to maintain direction
            
            elif strategy == 4:  # Circular Motion
                axis = circular_directions[group]
                angle = 0.1  # Constant angular step size
                rotation_matrix = rotation_matrix_around_axis(axis, angle)
                velocities[i] = np.dot(rotation_matrix, velocities[i])
            
            # Update position
            positions[i] += velocities[i]

        return positions

    # Simulation
    for t in range(timesteps):
        positions = update_positions(positions, velocities, group_assignments, noise_level)
        
        # Store positions and cluster assignments for each point at this timestep
        for i, pos in enumerate(positions):
            trajectory_data[i, t, :3] = pos  # Store x, y, z
            trajectory_data[i, t, 3] = group_assignments[i]  # Store cluster ID
        
        # Change group memberships periodically
        if t % group_change_interval == 0:
            group_assignments = np.random.randint(0, num_groups, num_points)

    return trajectory_data