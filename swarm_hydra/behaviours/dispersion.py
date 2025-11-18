import numpy as np
from collections import namedtuple
import logging

from swarm_hydra.entry_point import *
from swarm_hydra.behaviours.utils_behaviours import handling_box_limits
from swarm_hydra.behaviours.potential_field import (
    get_distances_and_angles,
    compute_repulsive_field,
)
from swarm_hydra.metrics.spatial_metrics import shift

# A logger for this file
log = logging.getLogger(__name__)
# log.setLevel(logging.DEBUG)


# Define Boid data structure.
Boids_Dispersion = namedtuple("Boids_Dispersion", ["X", "X_dot"])


def initialize_boids_disperation(
    box_size: float, boid_count: int, dim: int, seed: int, starting_speed: float
) -> Boids_Dispersion:
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
        A namedtuple `Boids_Dispersion`
    """
    reset_seeds(seed)
    _x = box_size * np.random.uniform(0, 1, (boid_count, dim))
    _x_dot = np.random.uniform(0, 1, (boid_count, dim))
    _x_dot = _x_dot / np.linalg.norm(_x_dot, axis=-1)[:, np.newaxis] * starting_speed

    return Boids_Dispersion(X=_x, X_dot=_x_dot)


###############################################################################
# Repulsive potential field from Hamann et al. 2022
###############################################################################


def repulsive_field_dispersion_force(positions: np.ndarray, params: dict) -> tuple:
    """Compute the repulsive potential and force field from agents."""
    dists, angles = get_distances_and_angles(
        positions, {k: params[k] for k in ["box_size", "walls_b"]}
    )

    repulsive_force = np.array(
        [
            compute_repulsive_field(dists[i], angles[i], params)
            for i in range(dists.shape[0])
        ]
    )

    return repulsive_force, dists, angles


def repulsive_field_dispersion_step(
    state: Boids_Dispersion, params: dict
) -> Boids_Dispersion:
    """"""
    X, X_dot = state.X, state.X_dot

    repulsive_force, dists, angles = repulsive_field_dispersion_force(X, params)
    total_force = repulsive_force * params["repul_w"]
    if params[
        "walls_b"
    ]:  # Handling bounded box case w/ a repulsive force near the boundaries
        total_force = handling_box_limits(
            total_force, X, {k: params[k] for k in ["box_init", "box_size", "max_dist"]}
        )

    # Update positions and orientations
    x_dot_upd = np.clip(
        X_dot + np.clip(total_force, -params["f_lim"], params["f_lim"]),
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
            params["interm_data_names"][3]: total_force,
            params["interm_data_names"][4]: dists,
            params["interm_data_names"][5]: angles,
        }
        return Boids_Dispersion(X=x_upd, X_dot=x_dot_upd), interm_data
    else:
        return Boids_Dispersion(X=x_upd, X_dot=x_dot_upd)
