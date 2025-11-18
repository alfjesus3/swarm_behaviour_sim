import numpy as np
from collections import namedtuple
import logging

from swarm_hydra.entry_point import *
from swarm_hydra.behaviours.utils_behaviours import handling_box_limits
from swarm_hydra.behaviours.potential_field import (
    get_distances_and_angles,
    compute_attractive_field,
)
from swarm_hydra.metrics.spatial_metrics import shift

# A logger for this file
log = logging.getLogger(__name__)
# log.setLevel(logging.DEBUG)


# Define Boid data structure.
Boids_Aggregation = namedtuple("Boids_Aggregation", ["X", "X_dot"])


def initialize_boids_aggregation(
    box_size: float, boid_count: int, dim: int, seed: int, starting_speed: float
) -> Boids_Aggregation:
    """
    Initialize boids with random positions and orientations for the Vicsek model.
    Initializes boids with random positions and velocities for the Reynolds model.

    Args:
        box_size: Size of the simulation box.
        boid_count: Number of boids to initialize.
        dim: Dimensionality of the positions (e.g., 2D or 3D).
        seed: Random seed for reproducibility.
        starting_speed: Constant initial speed for boids.

    Returns:
        A namedtuple `Boids_Aggregation`
    """
    reset_seeds(seed)
    _x = box_size * np.random.uniform(0, 1, (boid_count, dim))
    _x_dot = np.random.uniform(0, 1, (boid_count, dim))
    _x_dot = _x_dot / np.linalg.norm(_x_dot, axis=-1)[:, np.newaxis] * starting_speed

    return Boids_Aggregation(X=_x, X_dot=_x_dot)


###############################################################################
# Attractive potential field from Hamann et al. 2022
###############################################################################


def probabilistic_aggregation_force(positions: np.ndarray, params: dict) -> tuple:
    """Compute the force field from agents."""
    dists, angles = get_distances_and_angles(
        positions, {k: params[k] for k in ["box_size", "walls_b"]}
    )

    # Compute attraction field force
    attractive_force = np.array(
        [
            compute_attractive_field(dists[i], angles[i], params)
            for i in range(dists.shape[0])
        ]
    )

    return attractive_force, dists, angles


def probabilistic_aggregation_step(
    state: Boids_Aggregation, params: dict
) -> Boids_Aggregation:
    """
    Implements the probabilistic aggregation strategy 4 using a potential field approach for attraction
    and repulsion based on distance thresholds.

    Parameters:
    - boids: `Boids_Aggregation` namedtuple with the swarm configuration state.
    - params: dict -> Dictionary of params for probabilistic aggregation.

    Returns:
    - A namedtuple `Boids_Aggregation`
    """
    X, X_dot = state.X, state.X_dot

    attraction_force, dists, angles = probabilistic_aggregation_force(X, params)
    movement_force = attraction_force * params["attra_w"]

    if params[
        "walls_b"
    ]:  # Handling bounded box case w/ a repulsive force near the boundaries
        movement_force = handling_box_limits(
            movement_force,
            X,
            {k: params[k] for k in ["box_init", "box_size", "max_dist"]},
        )

    # Update positions and orientations
    x_dot_upd = np.clip(
        X_dot + np.clip(movement_force, -params["f_lim"], params["f_lim"]),
        -params["v_lim"],
        params["v_lim"],
    )
    x_dot_upd = (
        x_dot_upd
        * params["v_const"]
        / (np.linalg.norm(x_dot_upd, axis=-1)[:, np.newaxis] + params["eps"])
    )  # normalizing to fixed `v_const` speed
    x_upd = X

    # Apply updates over inner_loop iterations for smoother dynamics
    for _ in range(params["inner_loop"]):
        x_upd = shift(
            x_upd,
            (params["dt"] / params["inner_loop"]) * (params["sim_speed"] * x_dot_upd),
            params["box_size"],
        )

    if params["store_interm_data"]:
        interm_data = {
            params["interm_data_names"][3]: movement_force,
            params["interm_data_names"][4]: dists,
            params["interm_data_names"][5]: angles,
        }
        return Boids_Aggregation(X=x_upd, X_dot=x_dot_upd), interm_data
    else:
        return Boids_Aggregation(X=x_upd, X_dot=x_dot_upd)
