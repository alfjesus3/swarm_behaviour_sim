'''
import hydra
import logging

import numpy as np
import scipy
import matplotlib.pyplot as plt
import networkx as nx


from swarm_hydra.behaviours.flocking import \
    Boids_Reynolds, Boids_Vicsek, original_reynolds_force, \
        original_vicsek_force, from_vec_to_angle

# A logger for this file
log = logging.getLogger(__name__)


def construct_boids_k_tree(boids_state, w_assignment_fn, params):
    """
    Constructs a k-tree structure for a swarm of Boids.
    
    Parameters:
    ...
    
    Returns:
    - k_tree_edges (list of tuples): Edges of the k-tree representing the boid neighborhood structure.
    - spatial_weights (list of floats): Weights associated with each edge in the k-tree.
    """
    boids_positions, boids_velocities = boids_state.X, boids_state.X_dot
    
    num_boids = boids_positions.shape[0]
    
    # Build k-NN graph using a KDTree
    tree = scipy.spatial.KDTree(boids_positions)
    # d_neighbours, neighbor_indices = tree.query(boids_positions, distance_upper_bound=params['d_align'])
    # Since the tree.query() doesn't give the results as expected ...
    neighbor_indices = [
        tree.query_ball_point(boids_positions[i], p=2, r=params['d_align']) \
            for i in range(num_boids)]
    
    k_tree_edges = []
    spatial_weights = []
    
    for i in range(num_boids):
        for j in neighbor_indices[i]:
            if j !=i:  # Ignore self (???? index)
                pair_positions = np.array([boids_positions[i], boids_positions[j]])
                pair_velocities = np.array([boids_velocities[i], boids_velocities[j]])
                pair_boids = Boids_Reynolds(X=pair_positions, X_dot=pair_velocities) if \
                    isinstance(boids_state, Boids_Reynolds) else \
                        Boids_Vicsek(X=pair_positions, X_dot=pair_velocities, Theta=from_vec_to_angle(pair_velocities))
                
                # Compute edge weight
                weight = np.sum(
                    np.linalg.norm(
                        hydra.utils.instantiate(w_assignment_fn, pair_boids, params)[0], axis=-1
                    )
                )
                
                k_tree_edges.append((i, j))
                spatial_weights.append(weight)
    
    return k_tree_edges, spatial_weights


def plot_kinematic_tree_and_swarm(edges:list, weights: list, node_positions: np.ndarray, out_name: str):
    """
    Plots:
    1. Kinematic tree structure using `neato` layout.
    2. Swarm pose using **absolute 2D positions** (instead of auto-layout).

    Parameters:
    - edges: List of tuples representing kinematic tree edges [(parent, child), ...].
    - weights: List of spatial weight assignments (one per edge).
    - node_positions: Dictionary mapping node index → (x, y) absolute position.
    """
    
    # Normalize edge weights for visualization
    min_w, max_w = min(weights), max(weights)
    norm_weights = [(w - min_w) / (max_w - min_w) for w in weights] if max_w > min_w else [1] * len(weights)

    # Create an **undirected** graph
    G = nx.Graph()
    for (u, v), w in zip(edges, norm_weights):
        G.add_edge(u, v, weight=w)  # Assign weights to edges

    # --- Kinematic Tree Layout ---
    pos_tree = nx.nx_agraph.graphviz_layout(G, prog="neato", args='-Gnodesep=1.5')  # Auto-arrange

    # --- Swarm Pose Layout (Using Given Absolute Positions) ---
    pos_swarm = {i: (node_positions[i, 0], node_positions[i, 1]) for i in range(node_positions.shape[0])}  

    # --- Plot the Kinematic Tree ---
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    nx.draw(G, pos_tree, with_labels=True, node_size=80, node_color="lightblue", 
            edge_color="gray", font_size=7, width=1)
    plt.title("Kinematic Tree")

    # --- Plot the Swarm Pose (Using Given Absolute Positions) ---
    plt.subplot(1, 2, 2)
    nx.draw(G, pos_swarm, with_labels=True, node_size=80, node_color="lightgreen", 
            edge_color="black", width=norm_weights, font_size=7)
    plt.title("Swarm Pose (Nodes at boids' absolute positions)")

    # --- Determine Axis Limits for Reference Frame ---
    x_min, x_max = node_positions[:, 0].min(), node_positions[:, 0].max()
    y_min, y_max = node_positions[:, 1].min(), node_positions[:, 1].max()
    x_range, y_range = x_max - x_min, y_max - y_min

    # --- Add Reference Frame Axes from (0,0) ---
    ax = plt.gca()
    ax.set_xlim(x_min - 0.1 * x_range, x_max + 0.1 * x_range)  # Extend limits slightly
    ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)

    arrow_length = max(x_range, y_range) * 0.1  # Set arrow size as 10% of range
    plt.arrow(0, 0, arrow_length, 0, head_width=0.02 * y_range, head_length=0.03 * x_range, fc='red', ec='red', linewidth=2)
    plt.arrow(0, 0, 0, arrow_length, head_width=0.02 * x_range, head_length=0.03 * y_range, fc='blue', ec='blue', linewidth=2)

    # --- Add Labels for Axes ---
    plt.text(0.05 * x_range, -0.05 * y_range, "X", color='red', fontsize=12, fontweight='bold')
    plt.text(-0.05 * x_range, 0.05 * y_range, "Y", color='blue', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(out_name, bbox_inches='tight')
    log.info(f"Image saved as: {out_name}")
'''

'''
def compute_spatial_component(demo_sequence, imit_sequence, k_tree_edges, spatial_weights):
    """
    Computes the spatial component of the pose metric between two sequences of poses.
    
    Parameters:
    - demo_sequence (list of tuples): List of positions (e.g., joint coordinates) for the demonstrator.
    - imit_sequence (list of tuples): List of positions (e.g., joint coordinates) for the imitator.
    - k_tree_edges (list of tuples): Edges of the k-tree structure that connect the poses (indexed by joint pairs).
    - spatial_weights (list): List of weights to apply to the edges in the k-tree for weighting the dissimilarities.

    Returns:
    - spatial_distance (float): The total spatial dissimilarity measure.
    """
    spatial_distances = []

    # Iterate over the k-tree edges
    for edge, weight in zip(k_tree_edges, spatial_weights):
        demo_edge_start, demo_edge_end = demo_sequence[edge[0]], demo_sequence[edge[1]]
        imit_edge_start, imit_edge_end = imit_sequence[edge[0]], imit_sequence[edge[1]]
        
        # Calculate the instantaneous slopes for each pose (delta x, delta y for the edges)
        demo_slope = np.array(demo_edge_end) - np.array(demo_edge_start)
        imit_slope = np.array(imit_edge_end) - np.array(imit_edge_start)
        
        # Calculate the squared difference of the slopes
        slope_diff = np.sum((demo_slope - imit_slope) ** 2)
        
        # Weight the squared difference based on spatial_weights
        weighted_slope_diff = weight * slope_diff
        spatial_distances.append(weighted_slope_diff)

    # Return the sum of the weighted dissimilarities as the spatial distance
    return np.sum(spatial_distances)

# Example usage:
demo_sequence = [(0, 0), (1, 1), (2, 2), (3, 3)]  # Example positions for the demonstrator
imit_sequence = [(0, 0), (1, 1), (2, 2), (3, 3)]  # Example positions for the imitator
k_tree_edges = [(0, 1), (1, 2), (2, 3)]  # Example edges in the k-tree (pairs of indices in the sequence)
spatial_weights = [1.0, 1.5, 1.0]  # Example weights for each edge

# Compute the spatial component
spatial_distance = compute_spatial_component(demo_sequence, imit_sequence, k_tree_edges, spatial_weights)
print(f"Spatial Distance: {spatial_distance}")


or 


def compute_spatial_component(demo_sequence, imit_sequence, k_tree_edges, weights):
    """
    Computes the spatial component of the pose metric using the summed squared 
    difference of instantaneous slopes over k-tree edges.

    Parameters:
        demo_sequence (np.array): Demonstrator's pose sequence (N x D, where N is the number of points, D is the dimension).
        imit_sequence (np.array): Imitator's pose sequence (N x D).
        k_tree_edges (list of tuples): List of edges representing the k-tree structure.
        weights (list of float): Weights assigned to each edge.

    Returns:
        float: The computed spatial distance.
    """
    spatial_distance = 0.0

    for (i, j), weight in zip(k_tree_edges, weights):
        # Compute edge vectors
        demo_edge_vector = demo_sequence[j] - demo_sequence[i]
        imit_edge_vector = imit_sequence[j] - imit_sequence[i]

        # Compute instantaneous slope (gradient)
        demo_slope = np.linalg.norm(demo_edge_vector) / (np.linalg.norm(demo_sequence[i]) + 1e-8)  # Avoid division by zero
        imit_slope = np.linalg.norm(imit_edge_vector) / (np.linalg.norm(imit_sequence[i]) + 1e-8)

        # Compute squared difference of slopes
        slope_difference = (demo_slope - imit_slope) ** 2

        # Apply weight and accumulate
        spatial_distance += weight * slope_difference

    return spatial_distance
'''


# # TODO finalize the full vision of the "pose metric formulation with open chain kinematic trees (Amit R. et al.)"
# # (Using ChatGPT proposed code ...)

# import numpy as np
# from scipy.spatial.distance import euclidean
# from fastdtw import fastdtw

# def compute_spatial_component(demo_sequence, imit_sequence, weights):
#     spatial_distances = []
#     for demo_point, imit_point, weight in zip(demo_sequence, imit_sequence, weights):
#         distance = euclidean(demo_point, imit_point)
#         weighted_distance = weight * distance
#         spatial_distances.append(weighted_distance)
#     return np.sum(spatial_distances)

# def compute_temporal_component(demo_sequence, imit_sequence, temporal_weights):
#     distance, path = fastdtw(demo_sequence, imit_sequence, dist=euclidean)
#     weighted_distance = distance * temporal_weights
#     return weighted_distance

# def pose_metric(demo_sequence, imit_sequence, spatial_weights, temporal_weight=1.0):
#     spatial_component = compute_spatial_component(demo_sequence, imit_sequence, spatial_weights)
#     temporal_component = compute_temporal_component(demo_sequence, imit_sequence, temporal_weight)
#     total_metric = spatial_component + temporal_component
#     return total_metric
