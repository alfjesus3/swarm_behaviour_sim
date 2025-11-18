import numpy as np
from collections import namedtuple
import logging

from scipy.stats import levy_stable

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
Boids_BallisticRW = namedtuple("Boids_BallisticRW", ["X", "X_dot"])
Boids_BrownianRW = namedtuple("Boids_BrownianRW", ["X", "X_dot", "Counter"])


def initialize_boids_random_walk(
    box_size: float,
    boid_count: int,
    dim: int,
    seed: int,
    starting_speed: float,
    boid_type: str,
    counter_distrib_args: tuple,
) -> Boids_BallisticRW | Boids_BrownianRW:
    """
    Initialize boids with random positions and orientations for the Vicsek model.
    Initializes boids with random positions and velocities for the Reynolds model.

    Args:
        box_size: Size of the simulation box.
        boid_count: Number of boids to initialize.
        dim: Dimensionality of the positions (e.g., 2D or 3D).
        seed: Random seed for reproducibility.
        starting_speed: Constant initial speed for boids.
        boid_type: Either  of [ballistic, brownian]

    Returns:
        A namedtuple `Boids_BallisticRW` or `Boids_BrownianRW`
    """
    reset_seeds(seed)
    _x = box_size * np.random.uniform(0, 1, (boid_count, dim))
    _x_dot = np.random.uniform(0, 1, (boid_count, dim))
    _x_dot = _x_dot / np.linalg.norm(_x_dot, axis=-1)[:, np.newaxis] * starting_speed

    if boid_type == "ballistic":
        return Boids_BallisticRW(X=_x, X_dot=_x_dot)
    elif boid_type == "brownian":
        _counter = sample_levy_step_length(
            mu=counter_distrib_args[0],
            scale=counter_distrib_args[1],
            size=boid_count,
        ).astype(np.int32)
        return Boids_BrownianRW(X=_x, X_dot=_x_dot, Counter=_counter)
    else:
        raise NotImplementedError


###############################################################################
# Random Walk (Ballistic Motion) Strategy from M.Birattari et al. 2019
###############################################################################


def ballistic_random_walk_force(positions: np.ndarray, params: dict) -> tuple:
    """Compute the repulsive potential force field and drive straight direction for all agents."""
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


def ballistic_random_walk_step(
    state: Boids_BallisticRW, params: dict
) -> Boids_BallisticRW:
    """"""
    X, X_dot = state.X, state.X_dot

    repulsive_force, dists, angles = ballistic_random_walk_force(X, params)
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
        return Boids_BallisticRW(X=x_upd, X_dot=x_dot_upd), interm_data
    else:
        return Boids_BallisticRW(X=x_upd, X_dot=x_dot_upd)


###############################################################################
# Random Walk (Brownian Motion) Strategy from M.Birattari et al. 2019
###############################################################################


def sample_wrapped_cauchy_angle(rho, size=1):
    """
    Sample turning angles from wrapped Cauchy distribution
    f_w(θ; μ, ρ) = (1/(2π)) * (1-ρ²)/(1+ρ²-2ρcos(θ-μ))

    Args:
        rho: Concentration parameter (0 ≤ ρ < 1)
        size: Number of samples

    Returns:
        Array of angles in radians [-π, π]
    """
    if rho == 0:
        return np.random.uniform(-np.pi, np.pi, size=size)

    # Use inverse transform sampling for wrapped Cauchy
    u = np.random.uniform(0, 1, size=size)

    # For wrapped Cauchy with μ=0:
    # Inverse CDF: θ = 2 * arctan((1+ρ)/(1-ρ) * tan(π(u-0.5)))
    angles = 2 * np.arctan((1 + rho) / (1 - rho) * np.tan(np.pi * (u - 0.5)))

    return angles


def sample_levy_step_length(mu, scale=1.0, size=1):
    """
    Sample step lengths according to power law distribution P_μ(δ) ∼ δ^-(μ+1)

    For Brownian motion (μ≥2): finite second moment → normal distribution (CLT)
    For Lévy walks (μ<2): infinite variance → heavy-tailed distribution

    Args:
        mu: Power law exponent
               - μ ≥ 2: finite variance (Brownian-like)
               - μ < 2: infinite variance (Lévy walk)
        scale: Scale parameter controlling characteristic step size
        size: Number of samples

    Returns:
        Array of step lengths (always positive)
    """
    if mu >= 2.0:
        # Finite second moment case - use normal distribution
        # This complies with "tends to normal distribution according to CLT"
        step_lengths = np.abs(np.random.normal(loc=0, scale=scale, size=size))

        # Ensure minimum step size
        min_step = scale * 0.01
        step_lengths = np.maximum(step_lengths, min_step)
    else:
        raise NotImplementedError

        # Infinite variance case (μ < 2) - use power law sampling
        # P(δ) ∼ δ^-(μ+1) for δ ≥ δ_min

        delta_min = scale * 0.01  # Minimum step length to avoid singularity
        u = np.random.uniform(0, 1, size=size)

        # Avoid u=1 to prevent infinite values
        u = np.clip(u, 1e-10, 0.999)

        if abs(mu - 1.0) < 1e-6:
            # Special case for μ ≈ 1: P(δ) ∼ δ^-2 (Cauchy-like tail)
            step_lengths = delta_min / u
        else:
            # General power law inverse transform: δ = δ_min * (1-u)^(-1/μ)
            step_lengths = delta_min * np.power(1 - u, -1 / alpha)

        # Apply reasonable upper cutoff to prevent extremely large steps
        max_step = scale * 100
        step_lengths = np.clip(step_lengths, delta_min, max_step)

    return step_lengths


def brownian_random_walk_force(
    positions: np.ndarray, counter: np.ndarray, params: dict
) -> tuple:
    """
    Adapted from pag. 187 (198) from "Random Walks in Swarm Robotics: An Experiment with Kilobots"
    Brownian motion with Lévy walk step-length distribution and wrapped Cauchy turning angles.

    Expected params:
        - alpha: Lévy exponent (0 < μ ≤ 2, μ=2 gives Brownian motion)
        - rho: Wrapped Cauchy concentration (0 ≤ ρ < 1, ρ=0 gives uniform angles)
        - step_scale: Scale parameter for step lengths
        - rand_w: Weight for random component
        - repul_w: Weight for repulsive component
        - box_size, walls_b: For distance/angle calculations
    """
    dists, angles = get_distances_and_angles(
        positions, {k: params[k] for k in ["box_size", "walls_b"]}
    )

    zero_counter_mask = counter == 0
    n_particles = positions.shape[0]

    # Sample turning angles from wrapped Cauchy distribution
    turning_angles = sample_wrapped_cauchy_angle(rho=params["rho"], size=n_particles)

    # Convert to Cartesian force components
    random_direction_force = (
        np.column_stack(
            [
                params["max_long_vel"] * np.cos(turning_angles),
                params["max_rot_vel"] * np.sin(turning_angles),
            ]
        )
        * zero_counter_mask[:, np.newaxis]
    )

    repulsive_force = np.array(
        [
            compute_repulsive_field(dists[i], angles[i], params)
            for i in range(dists.shape[0])
        ]
    )

    return (
        random_direction_force * params["rand_w"] + repulsive_force * params["repul_w"],
        dists,
        angles,
    )


def brownian_random_walk_step(
    state: Boids_BrownianRW, params: dict
) -> Boids_BrownianRW:
    """"""
    X, X_dot, Counter = state.X, state.X_dot, state.Counter

    Counter -= 1 * (Counter != 0)  # Degree the Counter

    total_force, dists, angles = brownian_random_walk_force(X, Counter, params)
    counter_upd = Counter + (
        sample_levy_step_length(
            mu=params["mu"], scale=params["beta_scale"], size=X.shape[0]
        ).astype(np.int32)
        * (Counter == 0)
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
            params["interm_data_names"][4]: total_force,
            params["interm_data_names"][5]: dists,
            params["interm_data_names"][6]: angles,
        }
        return (
            Boids_BrownianRW(X=x_upd, X_dot=x_dot_upd, Counter=counter_upd),
            interm_data,
        )
    else:
        return Boids_BrownianRW(X=x_upd, X_dot=x_dot_upd, Counter=counter_upd)
