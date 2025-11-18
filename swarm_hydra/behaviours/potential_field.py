import logging

import numpy as np

from swarm_hydra.entry_point import *
from swarm_hydra.metrics.spatial_metrics import distance_fn, periodic_displacement

# A logger for this file
log = logging.getLogger(__name__)
# log.setLevel(logging.DEBUG)


###############################################################################
# Potential Fields modelling from ROS2SWARM github repo
#   Adapted from
#       https://github.com/ROS2swarm/ROS2swarm/blob/master/src/ros2swarm/ros2swarm/utils/scan_calculation_functions.py
###############################################################################


def get_distances_and_angles(positions, params):
    """
    Compute distances and angles between and for all boids.
    """
    # Compute pairwise displacements
    displacements = np.subtract(
        positions[:, np.newaxis], positions
    )  # Shape: [n_boids, n_boids, spatial_dim]

    # Apply periodic boundary conditions to displacements if no walls
    if not params["walls_b"]:
        displacements = periodic_displacement(displacements, params["box_size"])

    # Compute distances using the corrected displacements
    distances = np.sqrt(np.sum(displacements**2, axis=-1))

    # Compute angles using corrected displacements
    angles = np.arctan2(displacements[:, :, 1], displacements[:, :, 0])

    return distances, angles


def adjust_ranges(ranges: np.ndarray, min_range: float, max_range: float) -> np.ndarray:
    """
    Adjust all ranges:
    - Values over max_range are set to max_range.
    - Values under min_range are set to 0.0.
    """
    ranges = np.clip(ranges, min_range, max_range)  # Clamp to [min_range, max_range]
    ranges[ranges < min_range] = 0.0  # Set values below min_range to 0.0
    return ranges


def calculate_vectors_from_normed_ranges(ranges, angles):
    """
    Calculate a vector for each measured range in `ranges`.
    Only considers ranges < 1.0, assuming they are normalized.
    """
    mask = ranges < 1.0
    pol2cart = lambda r, theta: np.array([r * np.cos(theta), r * np.sin(theta)])
    return np.array(
        [pol2cart(ranges[i], angles[i]) for i in range(len(ranges)) if mask[i]]
    )


def calculate_vector_normed(ranges, angles):
    """
    Compute the final direction vector from sensor ranges and angles.
    """
    vectors = calculate_vectors_from_normed_ranges(ranges, angles)
    vector = np.sum(vectors, axis=0) if vectors.size > 0 else np.array([0.0, 0.0])
    vector *= -1  # Flip the direction of the given vector
    return vector


def create_normed_twist_message(
    vector, max_translational_velocity, max_rotational_velocity
):
    """
    Normalize the vector and scale it to match the max velocities.
    """
    norm = np.linalg.norm(vector)
    if norm == 0:
        return np.array([0.0, 0.0])

    vector = vector / norm  # Normalize the vector
    linear_velocity = max_translational_velocity * vector[0]
    angular_velocity = max_rotational_velocity * np.arcsin(
        vector[1] / np.hypot(vector[0], vector[1])
    )

    return np.array([float(linear_velocity), float(angular_velocity)])


def compute_attractive_field(
    distances: np.ndarray, angles: np.ndarray, params: dict
) -> np.ndarray:
    """"""
    ranges = adjust_ranges(distances, params["min_dist"], params["max_dist"])

    # Maps all range values to the interval [0,1] using a linear function.
    #   The closer the value is to max_range, the **less important** it is.
    ranges = 1 - (ranges / params["max_dist"])

    vector = calculate_vector_normed(ranges, angles)

    direction = create_normed_twist_message(
        vector, params["max_long_vel"], params["max_rot_vel"]
    )

    # if NaN then create stop twist message
    if np.isnan(direction[0]):
        log.error(
            "calculated Twist message contains NaN value, adjusting to a stop message"
        )
        direction = np.array([0.0, 0.0])

    return direction


def compute_repulsive_field(
    distances: np.ndarray, angles: np.ndarray, params: dict
) -> np.ndarray:
    """"""
    ranges = adjust_ranges(distances, params["min_dist"], params["max_dist"])

    # Map all ranges to the interval [0,1] using a linear function.
    #   As nearer the range is to max_range as __more__ important it is.
    #       Allowed input range for ranges is [0,max_range]
    ranges = ranges / params["max_dist"]

    vector = calculate_vector_normed(ranges, angles)
    vector *= -1  # Flip the direction of the given vector

    direction = create_normed_twist_message(
        vector, params["max_long_vel"], params["max_rot_vel"]
    )

    # if NaN then create stop twist message
    if np.isnan(direction[0]):
        log.error(
            "calculated Twist message contains NaN value, adjusting to a stop message"
        )
        direction = np.array([0.0, 0.0])

    return direction
