import numpy as np
import logging
from collections import Counter
import scipy

from swarm_hydra.metrics.utils_metrics import *

# A logger for this file
log = logging.getLogger(__name__)


def shift(positions: np.ndarray, delta: np.ndarray, box_size: float) -> np.ndarray:
    """
    Apply periodic boundary conditions.

    Args:
        positions (np.ndarray): Boids' positions; shape = [n_boids, spatial_dim].
        delta (np.ndarray): Position changes; shape = [n_boids, spatial_dim].
        box_size (float): Size of the simulation box.

    Returns:
        np.ndarray: Updated positions with periodic boundary conditions applied.
    """
    return (positions + delta) % box_size


def periodic_displacement(dx: np.ndarray, box_size: float) -> np.ndarray:
    """
    Calculate displacement considering periodic boundary conditions.

    Args:
        dx (np.ndarray): Raw displacement vector(s)
        box_size (float): Size of the simulation box

    Returns:
        np.ndarray: Corrected displacement considering wrap-around
    """
    # Find the shortest path considering wrap-around
    dx = dx - box_size * np.round(dx / box_size)
    return dx


def distance_fn(
    x: np.ndarray, y: np.ndarray, walls_b: bool, box_size: float
) -> np.ndarray:
    """
    Computes distances.

    Args:
        x (np.ndarray): First set of positions; shape = [..., spatial_dim].
        y (np.ndarray): Second set of positions; shape = [..., spatial_dim].
        walls_b (bool): If True, use classical distance computation.
            If False, account for periodic boundary conditions.
        box_size (float): Size of the simulation box.

    Returns:
        np.ndarray: Matrix of distances; shape = [...].
    """
    dx = np.subtract(x, y)

    if not walls_b:
        # Apply periodic boundary conditions to displacements
        dx = periodic_displacement(dx, box_size)

    dx = np.sum(dx**2, axis=-1)
    dists = np.sqrt(dx, where=(dx > 0), out=np.zeros_like(dx))

    return dists


def sort_agents_by_proximity_to_origin(
    agents_pos: np.array, walls_b: bool, box_size: float
) -> tuple[np.array, np.array]:
    """"""
    dists_to_origin = distance_fn(
        agents_pos, np.zeros_like(agents_pos), walls_b, box_size
    )
    # idxs = np.arange(dists_to_origin.shape[-1])  # Create [0, 1, ..., n-1]
    sorted_idxs = np.argsort(
        dists_to_origin, axis=-1
    )  # Get sorting indices for each row
    # Sort both arr and indices accordingly
    sorted_dists = np.take_along_axis(dists_to_origin, sorted_idxs, axis=-1)
    # sorted_idxs = np.take_along_axis(np.tile(idxs, (dists_to_origin.shape[0], 1)), sorted_idxs, axis=-1)
    # # Stack sorted values and their corresponding indices
    # sorted_idxs_w_dists = np.stack((sorted_idxs, sorted_dists), axis=-1)
    # return sorted_idxs_w_dists
    return sorted_idxs, sorted_dists


def preprocessing_data_for_metrics(boids_pos, other_arrays, params) -> tuple:
    """Arrays sorted in ascending order and with normalized values."""
    # Preprocessing the simulations' data for the metrics
    sorted_idxs, _ = sort_agents_by_proximity_to_origin(
        boids_pos, params["walls_b"], float(params["max_val"])
    )
    s_boids_pos = sort_array_by_indices(boids_pos, sorted_idxs)
    s_n_boids_pos = normalizing_quantities(
        s_boids_pos, q_type="abs_position", params=params
    )
    s_other_arr = sort_array_by_indices(other_arrays[0], sorted_idxs)
    s_n_other_arr = normalizing_quantities(
        s_other_arr, q_type="directional", params=params
    )

    return s_n_boids_pos, [s_n_other_arr]


def compute_mse_swarm_configuration(
    positions: np.ndarray, radius: float, walls_b: bool, box_size: float
) -> float:
    """Computes the "Mean Squared Error" (MSE) of inter-agent distances within a given radius."""
    num_agents = positions.shape[0]

    # Preallocate arrays for pairwise distances calculation
    i_indices, j_indices = np.triu_indices(num_agents, k=1)

    # Extract all pairs of positions based on the indices
    positions_i = positions[i_indices]
    positions_j = positions[j_indices]

    # Compute all distances at once
    all_distances = distance_fn(positions_i, positions_j, walls_b, box_size)

    # Filter distances within radius and square them
    valid_distances = all_distances[all_distances < radius]
    squared_distances = valid_distances**2

    # Return mean or 0 if no valid distances
    return np.mean(squared_distances) if len(squared_distances) > 0 else 0.0


def compute_convex_hull_area_swarm_configuration(positions: np.ndarray) -> float:
    """Compute the convex hull of a set of 2D points and return its area."""
    if len(positions) < 3:
        return 0.0  # A convex hull requires at least 3 points

    hull = scipy.spatial.ConvexHull(positions)  # Compute the convex hull

    # Extract and return the area
    return hull.area


def compute_polarisation_swarm_configuration(velocities: np.ndarray) -> float:
    """Computes the standard deviation of the velocity vectors."""
    return np.std(velocities)


def compute_cosine_similarity_measure(arr1: np.ndarray, arr2: np.ndarray) -> float:
    """
    (Reality check)
    d = 1 - c
    d + c = 1
    c = 1 - d
    """
    if arr1.shape != arr2.shape:
        raise ValueError("They should have the same shape...")

    return 1 - scipy.spatial.distance.cosine(arr1.flatten(), arr2.flatten())


def compute_euclidean_distance_measure(
    arr1: np.ndarray, arr2: np.ndarray, walls_b: bool, box_size: float
) -> None:
    """"""
    if arr1.shape != arr2.shape:
        raise ValueError("They should have the same shape...")

    return distance_fn(arr1.flatten(), arr2.flatten(), walls_b, box_size)


def compute_combined_state_count_measure(
    data1,
    data2,
    num_bins=3,
    threshold=0.01,
):
    """
    Computes the Combined State Count and Bray-Curtis dissimilarity between two system states.
    Retrieved from
        http://www.cmap.polytechnique.fr//~nikolaus.hansen/proceedings/2013/GECCO/proceedings/p199.pdf

    Args:
        data1, data2 (np.ndarray): Combined sensor-effector data arrays of shape
            (n_simulations, n_timesteps, swarm_size, num_features), representing
            compressed sensor values (e.g., front, left, right, back) and effector data.
        num_bins (int): Number of bins for discretization (default: 3, as per K=3).
        threshold (float): Frequency threshold for filtering states (default: 0.01, as per T=1%).

    Returns:
        tuple: (num_states1, num_states2, dissimilarity), where num_states1 and num_states2 are
            the number of filtered states, and similarity is "1" minus the Bray-Curtis dissimilarity.
    """

    def map_and_hash(data, num_bins):
        """
        Discretizes combined sensor-effector data and creates a hashable state representation.
        Sums state counts across all simulations and time steps.
        """
        n_simulations, n_timesteps, swarm_size, num_features = data.shape

        # Compute global min/max for normalization across all data
        data_flat = data.reshape(-1, num_features)
        v_min = data_flat.min(axis=0)  # Shape: (num_features,)
        v_max = data_flat.max(axis=0)  # Shape: (num_features,)

        state_counts = Counter()

        # Process each simulation and timestep separately (following Algorithm 1)
        for sim in range(n_simulations):
            for t in range(n_timesteps):
                for r in range(swarm_size):
                    # Read state for robot r
                    theta_r = data[sim, t, r, :]  # Shape: (num_features,)

                    # Discretize using paper's formula: floor((theta - theta_min) / (theta_max - theta_min) * (K-1))
                    theta_r_prime = np.floor(
                        (theta_r - v_min) / (v_max - v_min + 1e-10) * (num_bins - 1)
                    ).astype(int)
                    theta_r_prime = np.clip(
                        theta_r_prime, 0, num_bins - 1
                    )  # Ensure within bounds

                    # Hash the discretized state
                    state_tuple = tuple(theta_r_prime)

                    # Increment count normalized by swarm size (as per paper: +1/swarmsize)
                    state_counts[state_tuple] += 1.0 / swarm_size

        return dict(state_counts)

    def filter_unique_states(state_frequencies, threshold):
        """
        Filters states with frequencies below the threshold using paper's logic.
        """
        # Calculate total sum as per paper's filtering formula
        total_sum = sum(state_frequencies.values())
        threshold_value = total_sum * threshold

        # Filter states where count > total_sum * T (paper's filtering condition)
        return {
            state: freq
            for state, freq in state_frequencies.items()
            if freq > threshold_value
        }

    def bray_curtis_dissimilarity(state_frequencies1, state_frequencies2):
        """
        Computes Bray-Curtis dissimilarity between two state frequency distributions.
        """
        all_states = set(state_frequencies1.keys()).union(state_frequencies2.keys())
        dist1 = np.array([state_frequencies1.get(state, 0) for state in all_states])
        dist2 = np.array([state_frequencies2.get(state, 0) for state in all_states])
        return scipy.spatial.distance.braycurtis(dist1, dist2)

    # Compute state frequencies for both systems
    state_frequencies1 = map_and_hash(data1, num_bins)
    state_frequencies2 = map_and_hash(data2, num_bins)

    # Filter states based on the threshold
    filtered_states1 = filter_unique_states(state_frequencies1, threshold)
    filtered_states2 = filter_unique_states(state_frequencies2, threshold)

    # Compute Bray-Curtis dissimilarity
    dissimilarity = bray_curtis_dissimilarity(filtered_states1, filtered_states2)

    return len(filtered_states1), len(filtered_states2), dissimilarity


def compute_sampled_average_state_measure(
    data1, data2, num_time_windows, num_steps_sim, eps=1e-10
):
    """
    Computes the Sampled Average State (SAS) metric for two swarms.
    Retrieved from
        http://www.cmap.polytechnique.fr//~nikolaus.hansen/proceedings/2013/GECCO/proceedings/p199.pdf

    Parameters:
    - data1, data2: Arrays of shape (num_simulations, num_steps_sim, num_robots, num_features)
                    containing sensor-effector data (e.g., 4 compressed sensor values + effectors).
    - num_time_windows: Number of time windows (e.g., 1, 10, 50).
    - num_steps_sim: Total number of time steps per simulation.
    - eps: Small value to avoid division by zero during normalization.

    Returns:
    - sas: Manhattan distance between the averaged characterizations of the two swarms.
    """

    def compute_theta(data, num_time_windows, num_steps_sim):
        """
        Computes the characterization vector theta for a swarm.

        Parameters:
        - data: Array of shape (num_simulations, num_steps_sim, num_robots, num_features).
        - num_time_windows: Number of time windows.
        - num_steps_sim: Total number of time steps per simulation.

        Returns:
        - theta: Averaged characterization vector of shape (num_time_windows * num_features,).
        """
        num_simulations, num_time_steps, num_robots, num_features = data.shape
        window_size = (
            num_steps_sim // num_time_windows
        )  # Assumes num_steps_sim is divisible by num_time_windows
        theta_per_sim = []

        for sim in range(num_simulations):
            theta_sim = []
            for w in range(num_time_windows):
                start = w * window_size
                end = (
                    start + window_size if w < num_time_windows - 1 else num_steps_sim
                )  # Last window includes remaining steps
                window_data = data[
                    sim, start:end, :, :
                ]  # Shape: (window_size, num_robots, num_features)
                avg_state = (num_time_windows / num_steps_sim) * np.mean(
                    window_data, axis=(0, 1)
                )  # Shape: (num_features,)
                theta_sim.append(avg_state)
            theta_sim = np.concatenate(
                theta_sim
            )  # Shape: (num_time_windows * num_features,)
            theta_per_sim.append(theta_sim)

        # Average across simulations
        theta = np.mean(
            theta_per_sim, axis=0
        )  # Shape: (num_time_windows * num_features,)
        return theta

    # Validate input shapes
    num_simulations, num_time_steps, num_robots, num_features = data1.shape
    assert data1.shape == data2.shape, "Input data shapes must match."
    assert (
        num_time_steps == num_steps_sim
    ), "Number of time steps must equal num_steps_sim."

    # Compute global min and max for each feature across all data
    all_data = np.concatenate(
        [data1, data2], axis=0
    )  # Shape: (2*num_simulations, num_steps_sim, num_robots, num_features)
    v_min = np.min(all_data, axis=(0, 1, 2))  # Shape: (num_features,)
    v_max = np.max(all_data, axis=(0, 1, 2))  # Shape: (num_features,)

    # Normalize data globally
    norm_data1 = (data1 - v_min) / (
        v_max - v_min + eps
    )  # Shape: (num_simulations, num_steps_sim, num_robots, num_features)
    norm_data2 = (data2 - v_min) / (v_max - v_min + eps)

    # Compute characterization vector theta for each swarm
    theta1 = compute_theta(
        norm_data1, num_time_windows, num_steps_sim
    )  # Shape: (num_time_windows * num_features,)
    theta2 = compute_theta(
        norm_data2, num_time_windows, num_steps_sim
    )  # Shape: (num_time_windows * num_features,)

    # Compute Manhattan distance
    sas = np.sum(np.abs(theta1 - theta2))

    return sas


def compute_center_of_mass_swarm_configuration(positions: np.ndarray) -> np.ndarray:
    """
    Computes the center of mass according to
        https://arxiv.org/pdf/2209.01118
    """
    return tuple(np.mean(positions, axis=0))


def compute_maximum_swarm_shift_swarm_configuration(
    positions: np.ndarray, walls_b: bool, box_size: float
) -> float:
    """
    Computes the maximum swarm shift according to
        https://arxiv.org/pdf/2209.01118
    """
    num_agents = positions.shape[0]

    # Preallocate arrays for pairwise distances calculation
    i_indices, j_indices = np.triu_indices(num_agents, k=1)

    # Extract all pairs of positions based on the indices
    positions_i = positions[i_indices]
    positions_j = positions[j_indices]

    # Compute all distances at once
    all_distances = distance_fn(positions_i, positions_j, walls_b, box_size)

    # Return mean or 0 if no valid distances
    return np.max(all_distances)


def compute_swarm_mode_index_swarm_configuration(
    positions: np.ndarray, freq_dist_threshold: float = 0.5
) -> np.ndarray:
    """
    Computes the swarm mode index according to
        https://arxiv.org/pdf/2209.01118
    """
    # Compute center of mass for the current time step
    com = np.mean(positions, axis=0)  # Shape: (2,)

    # Compute swarm mode for x and y separately
    x_coords = positions[:, 0]
    y_coords = positions[:, 1]

    # Find the mode x and y by maximizing frequency
    x_mode = None
    y_mode = None
    max_freq_x = 0
    max_freq_y = 0

    # Unique x and y values with a small tolerance (e.g., 0.1 as per formula)
    unique_x = np.unique(x_coords)
    unique_y = np.unique(y_coords)

    for x in unique_x:
        freq = np.sum(np.abs(x_coords - x) < freq_dist_threshold)
        if freq > max_freq_x:
            max_freq_x = freq
            x_mode = x

    for y in unique_y:
        freq = np.sum(np.abs(y_coords - y) < freq_dist_threshold)
        if freq > max_freq_y:
            max_freq_y = freq
            y_mode = y

    # Swarm mode location
    swarm_mode = np.array([x_mode, y_mode])

    # Compute distance between center of mass and swarm mode
    distances = np.linalg.norm(com - swarm_mode)

    # Average distance over all time steps
    return distances


def compute_longest_path_swarm_configuration(positions: np.ndarray) -> float:
    """
    Note: It assumes the origin is (0, 0)

    Computes the longest path according to
        https://arxiv.org/pdf/2209.01118
    """
    return np.max(np.linalg.norm(positions, axis=1))


def compute_maximum_radius_swarm_configuration(positions: np.ndarray) -> float:
    """
    Note: It assumes the origin is (0, 0)

    Computes the maximum radius according to
        https://arxiv.org/pdf/2209.01118
    """
    # Compute center of mass for each timestep
    c_o_m = compute_center_of_mass_swarm_configuration(positions)
    # Compute distance from center of mass to each agent
    distances = np.linalg.norm(positions - c_o_m, axis=1)  # Shape: (n_agents)
    # Maximum distance acrsoss all agents
    return np.max(distances)


def compute_average_local_density_measure(
    positions: np.ndarray, radius: float, box_size: float, walls_b: bool
) -> float:
    """
    Computes the average local density according to
        https://arxiv.org/pdf/2209.01118
    """
    if positions.shape[0] <= 1:
        return 0.0  # Avoid division by zero

    # Compute pairwise distances
    # dist_matrix = distance_matrix(positions, positions)
    dist_matrix = distance_fn(positions[:, np.newaxis], positions, walls_b, box_size)

    # Count neighbors within the given radius (excluding self)
    neighbor_counts = np.sum((dist_matrix < radius) & (dist_matrix > 0.0), axis=1)

    return np.mean(neighbor_counts)  # Average number of neighbors


def compute_beta_index_measure(
    positions: np.ndarray, box_size: float, walls_b: bool
) -> float:
    """
    Computes the beta index according to
        https://arxiv.org/pdf/2209.01118
    """
    N = positions.shape[0]  # Number of boids
    if N <= 2:
        return 0.0  # Avoid division by zero

    if positions.ndim == 3:
        b = positions.shape[1]  # Number of boids
        # Compute pairwise distances
        # dist_matrix = distance_matrix(positions, positions)
        dist_matrix = distance_fn(
            positions[:, :, np.newaxis, :],
            positions[:, np.newaxis, :, :],
            walls_b,
            box_size,
        )
        # Compute D_avg per simulation (n,)
        D_avg = np.sum(dist_matrix, axis=(1, 2)) / (b * (b - 1))
        # Compute adjacency matrix: 1 if dist_matrix < D_avg, 0 otherwise
        adjacency_matrix = (dist_matrix < D_avg[:, np.newaxis, np.newaxis]).astype(int)
        # Compute total edges E (sum adjacency matrix, divide by 2 for undirected edges)
        E = np.sum(adjacency_matrix) / 2
        # Compute beta index: β = E / N (N = b boids)
        E = E / b
    elif positions.ndim == 2:
        # Compute pairwise distances
        # dist_matrix = distance_matrix(positions, positions)
        dist_matrix = distance_fn(
            positions[:, np.newaxis], positions, walls_b, box_size
        )
        # Compute the average distance across all unique pairs (excluding self-distances)
        D_avg = np.sum(dist_matrix) / (N * (N - 1))  # Average distance
        # Create adjacency matrix where edges exist if d_ij < D_avg
        adjacency_matrix = (dist_matrix < D_avg).astype(int)
        # Compute the total number of edges (sum of adjacency matrix / 2 to avoid double counting)
        E = np.sum(adjacency_matrix) / 2
        # Compute beta index: β = E / N
        E = E / N  # Beta index

    return E


def compute_average_nearest_neighbour_distance_measure(
    positions: np.ndarray,
) -> float:
    """
    Computes the average nearest neighbour distance according to
        https://arxiv.org/pdf/2209.01118
    """
    n_agents, _ = positions.shape
    nearest_distances = []
    for i in range(n_agents):
        # Compute distances from agent i to all other agents
        dists = np.linalg.norm(positions - positions[i], axis=1)
        # Exclude self (distance to itself is 0)
        dists[i] = np.inf
        # Find nearest neighbor distance
        nearest_dist = np.min(dists)
        nearest_distances.append(nearest_dist)

    # Sum distances for this time step
    return np.sum(nearest_distances)


def compute_collision_count_measure(
    positions: np.ndarray, radius: float, walls_b: bool, box_size: float
) -> int:
    """
    Computes the collision count according to https://ieeexplore.ieee.org/document/10216297
    """
    num_agents = positions.shape[0]

    # Preallocate arrays for pairwise distances calculation
    # And already accounts for `agent1--agent2` is the same as `agent2--agent1`
    i_indices, j_indices = np.triu_indices(num_agents, k=1)

    # Extract all pairs of positions based on the indices
    positions_i = positions[i_indices]
    positions_j = positions[j_indices]

    # Compute all distances at once
    all_distances = distance_fn(positions_i, positions_j, walls_b, box_size)

    # Count collisions (distances less than threshold)
    return int(np.sum(all_distances < (radius / 2.0)))


def compute_flock_density_measure(positions: np.ndarray) -> float:
    """
    Computes the flock density according to https://ieeexplore.ieee.org/document/10216297
    """
    num_agents = positions.shape[0]
    area = compute_convex_hull_area_swarm_configuration(positions)

    return num_agents / area


def compute_grouping_measure(
    positions: np.ndarray, walls_b: bool, box_size: float
) -> float:
    """
    Computes the grouping according to https://ieeexplore.ieee.org/document/10216297
    """
    num_agents = positions.shape[0]
    if num_agents <= 1:
        return 0.0

    # Calculate pairwise distance matrix
    dist_matrix = distance_fn(positions[:, np.newaxis], positions, walls_b, box_size)

    # For each boid, calculate average distance to all others
    separations = []
    for i in range(num_agents):
        # Sum distances to all other boids, excluding self (distance = 0)
        other_distances = np.concatenate([dist_matrix[i, :i], dist_matrix[i, i + 1 :]])
        avg_separation = np.mean(other_distances)
        separations.append(avg_separation)

    return np.mean(separations)


def compute_straggler_count_measure(
    positions: np.ndarray, radius: float, walls_b: bool, box_size: float
) -> int:
    """
    Computes the straggler_count according to https://ieeexplore.ieee.org/document/10216297
    """
    threshold_distance = radius / 2

    num_agents = positions.shape[0]
    if num_agents <= 1:
        return 0

    # Calculate pairwise distance matrix
    dist_matrix = distance_fn(positions[:, np.newaxis], positions, walls_b, box_size)

    # Count stragglers: boids with no neighbors within threshold
    stragglers = 0
    for i in range(num_agents):
        # Check if boid i has any neighbors within threshold (excluding itself)
        neighbors = (
            np.sum(dist_matrix[i, :] < threshold_distance) - 1
        )  # -1 to exclude self
        if neighbors == 0:
            stragglers += 1

    return stragglers


def compute_order_measure(velocities: np.ndarray) -> tuple:
    """
    Computes the order according to https://ieeexplore.ieee.org/document/10216297
    """
    if len(velocities) == 0:
        return 0.0

    # Normalize each velocity vector
    norms = np.linalg.norm(velocities, axis=1, keepdims=True)

    # Avoid division by zero
    norms = np.where(norms == 0, 1, norms)
    normalized_velocities = velocities / norms

    # Return the average normalized velocity
    return tuple(np.mean(normalized_velocities, axis=0))


def compute_subgroup_count_measure(
    positions: np.ndarray, radius: float, walls_b: bool, box_size: float
) -> int:
    """
    Computes the subgroup count according to https://ieeexplore.ieee.org/document/10216297
    """
    connection_threshold = radius
    num_agents = positions.shape[0]
    if num_agents <= 1:
        return 0

    # Calculate pairwise distance matrix
    dist_matrix = distance_fn(positions[:, np.newaxis], positions, walls_b, box_size)

    # Create adjacency matrix: True if robots are connected (within threshold)
    adjacency = dist_matrix < connection_threshold
    # Remove self-connections (diagonal)
    np.fill_diagonal(adjacency, False)

    # Implementation of Navarro-Matía algorithm for finding connected components
    unvisited = set(range(num_agents))  # S in the algorithm
    subgroup_count = 0

    while unvisited:  # WHILE S is not empty
        subgroup_count += 1  # i = i + 1

        # Pick any robot from unvisited set
        current_robot = next(iter(unvisited))  # rs = a robot in S

        # Initialize current subgroup and queue for BFS/DFS
        current_subgroup = {current_robot}  # Ri = Ri + {rs}
        unvisited.remove(current_robot)  # S = S - {rs}

        # Queue for exploring connected robots
        to_explore = [current_robot]

        # Explore all robots connected to current subgroup
        while to_explore:
            exploring_robot = to_explore.pop()

            # Check all remaining unvisited robots
            robots_to_remove = []
            for candidate_robot in unvisited:  # FOR each robot raux in S
                # Check if there's a direct connection (path of length 1)
                if adjacency[
                    exploring_robot, candidate_robot
                ]:  # IF there is path from rs to raux
                    current_subgroup.add(candidate_robot)  # Ri = Ri + {raux}
                    robots_to_remove.append(candidate_robot)
                    to_explore.append(candidate_robot)  # Add to exploration queue

            # Remove connected robots from unvisited set
            for robot in robots_to_remove:
                unvisited.remove(robot)  # S = S - {raux}

    return subgroup_count


def compute_diffusion_measure(
    interval_of_positions: np.ndarray, time_lag: float
) -> tuple[float, float]:
    """
    Computes the diffusion according to https://ieeexplore.ieee.org/document/10216297
    Note: it uses the standard/normal diffusion computation from section 2.1 in https://arxiv.org/pdf/1206.4434
    """
    if len(interval_of_positions) < time_lag + 1:
        raise ValueError(
            f"Need at least {time_lag + 1} time steps for time_lag={time_lag}"
        )

    if len(interval_of_positions[0]) == 0:
        return 0.0, 0.0

    T = len(interval_of_positions)  # Total time steps
    N = len(interval_of_positions[0])  # Number of boids

    # Verify all time steps have same number of boids
    for t, positions in enumerate(interval_of_positions):
        if len(positions) != N:
            raise ValueError(
                f"All time steps must have same number of boids. Step {t} has {len(positions)}, expected {N}"
            )

    # Calculate mean-square displacement according to equation (1)
    total_squared_displacement = 0.0

    # Sum over all time lags of duration t in interval [0, T-t]
    for t0 in range(T - time_lag):  # t0 = 0 to T-t-1
        positions_initial = interval_of_positions[t0]  # Ri(t0)
        positions_final = interval_of_positions[t0 + time_lag]  # Ri(t0 + t)

        # Sum over all N individuals
        for i in range(N):  # i = 1 to N
            # Calculate squared displacement for individual i
            displacement = (
                positions_final[i] - positions_initial[i]
            )  # Ri(t0+t) - Ri(t0)
            squared_displacement = np.sum(displacement**2)  # ||Ri(t0+t) - Ri(t0)||²

            total_squared_displacement += squared_displacement

    # Apply the normalization factors from equation (1)
    # δR²(t) = (1/(T-t)) * (1/N) * Σ Σ [squared displacements]
    mean_squared_displacement = total_squared_displacement / (T - time_lag) / N

    # Diffusion coefficient assuming normal diffusion (α = 1)
    # General: δR²(t) = 2d*D*t, where d is dimensionality
    dimensionality = interval_of_positions[0].shape[1]
    diffusion_coefficient = mean_squared_displacement / (2 * dimensionality * time_lag)

    return mean_squared_displacement, diffusion_coefficient


def compute_neighbor_shortest_distances_measure(
    positions: np.ndarray,
    arena_diameter: float,
    sort_descending: bool,
    walls_b: bool,
    box_size: float,
) -> np.ndarray:
    """
    Computes the nearest neighbor distances according to https://arxiv.org/abs/2301.06864
    """
    if positions.shape[0] <= 1:
        return np.array([])

    # Calculate pairwise distance matrix
    dist_matrix = distance_fn(positions[:, np.newaxis], positions, walls_b, box_size)

    # For each agent, find distance to nearest neighbor
    # Set diagonal to infinity to exclude self-distances
    np.fill_diagonal(dist_matrix, np.inf)
    nearest_neighbor_distances = np.min(dist_matrix, axis=1)

    # Apply the normalization formula: 10^(-2x/d)
    # where x is distance and d is arena diameter
    normalized_features = np.power(10, -2 * nearest_neighbor_distances / arena_diameter)

    scaled_features = normalized_features

    # Sort features as specified in the paper
    if sort_descending:
        # Sort in descending order (nearest neighbors get higher feature values)
        sorted_features = np.sort(scaled_features)[::-1]
    else:
        sorted_features = np.sort(scaled_features)

    return sorted_features
