import numpy as np
from collections import namedtuple
import logging

from swarm_hydra.entry_point import *
from swarm_hydra.behaviours.utils_behaviours import handling_box_limits
from swarm_hydra.metrics.spatial_metrics import shift, distance_fn

# A logger for this file
log = logging.getLogger(__name__)
# log.setLevel(logging.DEBUG)


# Define Boid data structure.
Boids_Vicsek = namedtuple("Boids_Vicsek", ["X", "X_dot", "Theta"])
Boids_Reynolds = namedtuple("Boids_Reynolds", ["X", "X_dot"])


def from_vec_to_angle(vels: np.ndarray) -> np.ndarray:
    """
    Converts velocity vectors to angles for Vicsek representation.
    (From vector to lenght-direction representation.)

    Args:
        vels (np.ndarray): Velocity vectors; shape = [n_boids, spatial_dim].

    Returns:
        np.ndarray: Angles representing velocities; shape = [n_boids].
    """
    return np.arctan2(
        vels[:, 1], vels[:, 0]
    )  # Note: already corrects for special quadrants II and III


def from_angle_to_vec(theta: np.ndarray) -> np.ndarray:
    """
    Converts angles to velocity vectors for Reynolds representation.
    (From lenght-direction to vector representation.)

    Args:
        theta (np.ndarray): Angles representing velocities; shape = [n_boids].

    Returns:
        np.ndarray: Velocity vectors; shape = [n_boids, spatial_dim].
    """
    return np.stack([np.cos(theta), np.sin(theta)]).T


def initialize_boids_flocking(
    box_size: float,
    boid_count: int,
    dim: int,
    seed: int,
    starting_speed: float,
    boid_type: str,
) -> Boids_Vicsek | Boids_Reynolds:
    """
    Initialize boids with random positions and orientations for the Vicsek model.
    Initializes boids with random positions and velocities for the Reynolds model.

    Args:
        box_size: Size of the simulation box.
        boid_count: Number of boids to initialize.
        dim: Dimensionality of the positions (e.g., 2D or 3D).
        seed: Random seed for reproducibility.
        starting_speed: Constant initial speed for boids.
        boid_type: Either  of [reynolds, vicsek]

    Returns:
        A namedtuple `Boids_Vicsek` or `Boids_Reynolds`
    """
    reset_seeds(seed)
    _x = box_size * np.random.uniform(0, 1, (boid_count, dim))
    _theta = np.random.uniform(0, 2 * np.pi, boid_count)
    _x_dot = starting_speed * from_angle_to_vec(_theta)

    if boid_type == "vicsek":
        return Boids_Vicsek(X=_x, X_dot=_x_dot, Theta=_theta)
    elif boid_type == "reynolds":
        return Boids_Reynolds(X=_x, X_dot=_x_dot)
    else:
        raise NotImplementedError


###############################################################################
# Original Reynolds flocking model (w/ minor adaptations)
###############################################################################


def original_reynolds_force(state: Boids_Reynolds, params: dict) -> tuple:
    """"""
    X, X_dot = state.X, state.X_dot  # n_boids, 2

    displ = np.subtract(X[:, np.newaxis], X)  # Pairwise displacements
    dists = distance_fn(X[:, np.newaxis], X, params["walls_b"], params["box_size"])

    #####
    # Alignment force
    #####
    dists_by_thres = dists / params["d_align"]
    vels_w_mask = (
        X_dot * ((dists_by_thres < 1.0) & (dists_by_thres != 0.0))[:, :, np.newaxis]
    )
    norm_vels_w_mask = vels_w_mask / (
        np.linalg.norm(vels_w_mask, axis=-1)[:, :, np.newaxis] + params["eps"]
    )  # since the L2 normalization uses all dimensions (hence axis=-1 for last dim)
    total_vels_by_thres = np.sum(norm_vels_w_mask, axis=1)
    total_neigh_by_thres = np.sum(
        (dists_by_thres < 1.0) & (dists_by_thres != 0.0), axis=1
    )
    align_res = total_vels_by_thres / (
        total_neigh_by_thres[:, np.newaxis] + params["eps"]
    )
    align_res = np.where(np.isfinite(align_res), align_res, 0)

    #####
    # Cohesion force
    #####
    dists_by_thres = dists / params["d_cohe"]
    displ_w_mask = (
        displ * ((dists_by_thres < 1.0) & (dists_by_thres != 0.0))[:, :, np.newaxis]
    )
    norm_displ_w_mask = displ_w_mask / (
        np.linalg.norm(displ_w_mask, axis=-1)[:, :, np.newaxis] + params["eps"]
    )
    total_displ_by_thres = -np.sum(norm_displ_w_mask, axis=1)
    # This has A minus since attraction_force += (neighbor - position) / np.linalg.norm(neighbor - position)
    #   instead of
    #       separation_force -= (neighbor - position) / np.linalg.norm(neighbor - position)
    total_neigh_by_thres = np.sum(
        (dists_by_thres < 1.0) & (dists_by_thres != 0.0), axis=1
    )
    cohe_res = total_displ_by_thres / (
        total_neigh_by_thres[:, np.newaxis] + params["eps"]
    )
    cohe_res = np.where(np.isfinite(cohe_res), cohe_res, 0)

    #####
    # Separation force
    #####
    dists_by_thres = dists / params["d_sepa"]
    displ_w_mask = (
        displ * ((dists_by_thres < 1.0) & (dists_by_thres != 0.0))[:, :, np.newaxis]
    )
    norm_displ_w_mask = displ_w_mask / (
        np.linalg.norm(displ_w_mask, axis=-1)[:, :, np.newaxis] + params["eps"]
    )
    subtracted_displ_by_thres = np.sum(norm_displ_w_mask, axis=1)
    # This has NO minus since attraction_force += (neighbor - position) / np.linalg.norm(neighbor - position)
    #   instead of
    #       separation_force -= (neighbor - position) / np.linalg.norm(neighbor - position)
    total_neigh_by_thres = np.sum(
        (dists_by_thres < 1.0) & (dists_by_thres != 0.0), axis=1
    )
    sepa_res = subtracted_displ_by_thres / (
        total_neigh_by_thres[:, np.newaxis] + params["eps"]
    )
    sepa_res = np.where(np.isfinite(sepa_res), sepa_res, 0)

    # Compute total force
    return (
        align_res * params["coef_alignment"]
        + cohe_res * params["coef_cohesion"]
        + sepa_res * params["coef_separation"],
        dists,
        displ_w_mask,
        vels_w_mask,
    )


def original_reynolds_step(
    state: Boids_Reynolds, params: dict
) -> Boids_Reynolds | tuple[Boids_Reynolds, dict]:
    """"""
    X, X_dot = state.X, state.X_dot  # n_boids, 2

    # Computing total force
    total_force, dists, displ_w_mask, vels_w_mask = original_reynolds_force(
        state, params
    )
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
    )  # normalizing to fixed `Vicsek` speed
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
            params["interm_data_names"][5]: (vels_w_mask - X_dot[:, np.newaxis, :])
            * (vels_w_mask != 0.0),
            params["interm_data_names"][6]: (displ_w_mask - X[:, np.newaxis, :])
            * (displ_w_mask != 0.0),
        }
        return Boids_Reynolds(X=x_upd, X_dot=x_dot_upd), interm_data
    else:
        return Boids_Reynolds(X=x_upd, X_dot=x_dot_upd)


###############################################################################
# Original Vicsek flocking model (w/ minor adaptations)
###############################################################################


def original_vicsek_force(state: Boids_Vicsek, params: dict) -> tuple:
    """"""
    X, theta = state.X, state.Theta

    dists = distance_fn(X[:, np.newaxis], X, params["walls_b"], params["box_size"])

    # Vicsek Alignment using proper circular averaging
    dists_by_thres = dists / params["d_align"]
    neighbors_mask = dists_by_thres < 1.0

    # Convert angles to unit vectors for proper averaging
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    # Calculate weighted sum of unit vectors
    sum_cos = np.sum(cos_theta[np.newaxis, :] * neighbors_mask, axis=1)
    sum_sin = np.sum(sin_theta[np.newaxis, :] * neighbors_mask, axis=1)

    # Convert back to angles
    align_res = np.arctan2(sum_sin, sum_cos)

    # Add noise
    align_res += np.random.uniform(-params["noise"], params["noise"], align_res.shape)

    # Normalize to desired range if needed
    align_res = np.mod(
        align_res, 2 * np.pi
    )  # or use np.mod(align_res + np.pi, 2*np.pi) - np.pi for [-π, π]

    return align_res, dists


def original_vicsek_step(
    state: Boids_Vicsek, params: dict
) -> Boids_Vicsek | tuple[Boids_Vicsek, dict]:
    """"""
    X, theta = state.X, state.Theta  # n_boids, 2

    # Computing total force
    align_res, dists = original_vicsek_force(state, params)
    if params[
        "walls_b"
    ]:  # Handling bounded box case w/ a repulsive force near the boundaries
        corresponding_vels = params["v_const"] * from_angle_to_vec(align_res)
        corresponding_vels = handling_box_limits(
            corresponding_vels,
            X,
            {k: params[k] for k in ["box_init", "box_size", "max_dist"]},
        )
        align_res = from_vec_to_angle(corresponding_vels)

    # Update positions and orientations
    theta_upd = align_res
    x_dot_upd = params["v_const"] * from_angle_to_vec(theta_upd)
    x_upd = X

    # Apply updates over inner_loop iterations for smoother dynamics
    for _ in range(params["inner_loop"]):
        x_upd = shift(
            x_upd,
            (params["dt"] / params["inner_loop"]) * (params["sim_speed"] * x_dot_upd),
            params["box_size"],
        )

    if params["store_interm_data"]:
        return Boids_Vicsek(X=x_upd, X_dot=x_dot_upd, Theta=theta_upd), {
            params["interm_data_names"][4]: theta_upd,
            params["interm_data_names"][5]: dists,
            params["interm_data_names"][6]: (align_res - theta),
        }
    else:
        return Boids_Vicsek(X=x_upd, X_dot=x_dot_upd, Theta=theta_upd)
