import hydra
from omegaconf import DictConfig, OmegaConf
import logging

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from copy import deepcopy

from swarm_hydra.entry_point import *
from swarm_hydra.behaviours.utils_behaviours import *
from swarm_hydra.behaviours.random_walk import *


# A logger for this file
log = logging.getLogger(__name__)


def unit_t1(output_dir: str, cfg: DictConfig) -> None:
    """
    Testing boids initialization, ballistic motion forces
        computation and forces rendering basics.
    """
    os.makedirs(output_dir + f"/unit_t1/", mode=0o777, exist_ok=True)

    # Test initialization
    boids_ballistic = initialize_boids_random_walk(
        cfg.behaviours.class_m.box_size,
        cfg.behaviours.class_m.boid_count,
        cfg.behaviours.class_m.dim,
        cfg.behaviours.class_m.seed,
        cfg.behaviours.class_m.init_speed,
        cfg.behaviours.class_m.model.name,
        (None, None),
    )
    assert boids_ballistic[0].shape == (
        cfg.behaviours.class_m.boid_count,
        cfg.behaviours.class_m.dim,
    ), "Should be the same shape."
    assert (
        boids_ballistic[1].shape == boids_ballistic[0].shape
    ), "Should be the same shape."
    assert np.isclose(
        np.mean(np.linalg.norm(boids_ballistic[1], axis=-1)),
        cfg.behaviours.class_m.init_speed,
        1e-3,
    ), "The inital speed must be constant."

    # Test random walk forces computation
    repulsive_f1, _, _ = ballistic_random_walk_force(
        boids_ballistic.X,
        {
            "box_size": cfg.behaviours.class_m.box_size,
            "walls_b": cfg.behaviours.class_m.walls_b,
            "min_dist": cfg.behaviours.class_m.model.min_dist,
            "max_dist": cfg.behaviours.class_m.model.max_dist,
            "max_long_vel": cfg.behaviours.class_m.max_long_vel,
            "max_rot_vel": cfg.behaviours.class_m.max_rot_vel,
            "max_abs_f": cfg.behaviours.class_m.max_abs_f,
        },
    )
    assert repulsive_f1.shape == (
        cfg.behaviours.class_m.boid_count,
        cfg.behaviours.class_m.dim,
    ), "Should be the same shape."

    # Visualization for random walk forces
    repulsive_f1_x, repulsive_f1_y = np.split(repulsive_f1, 2, axis=1)
    visualizing_boids_force(
        boids_ballistic.X,
        (
            (
                repulsive_f1_x,
                repulsive_f1_y,
                "red",
                cfg.behaviours.class_m.box_size // 3,
                "Repulsion force",
            ),
        ),
        output_dir + f"/unit_t1/boids_forces.png",
        "Ballistic Motion Boid Forces",
    )


def unit_t2(output_dir: str, state: dict, cfg: DictConfig) -> None:
    """
    Using the Ballistic Motion model for a single step
    """
    os.makedirs(output_dir + f"/unit_t2/", mode=0o777, exist_ok=True)

    cfg.behaviours.class_m.storing_data = True
    out_step1 = ballistic_random_walk_step(
        state["boids"],
        {
            "box_init": cfg.behaviours.class_m.box_init,
            "box_size": cfg.behaviours.class_m.box_size,
            "dt": cfg.behaviours.class_m.dt,
            "inner_loop": cfg.behaviours.class_m.inner_loop,
            "sim_speed": cfg.behaviours.class_m.sim_speed,
            "f_lim": cfg.behaviours.class_m.max_abs_f,
            "v_lim": cfg.behaviours.class_m.max_abs_sp,
            "max_speed": cfg.behaviours.class_m.max_abs_sp,
            "v_const": cfg.behaviours.class_m.init_speed,
            "eps": cfg.behaviours.class_m.eps,
            "walls_b": cfg.behaviours.class_m.walls_b,
            "store_interm_data": cfg.behaviours.class_m.storing_data,
            "interm_data_names": cfg.behaviours.class_m.model.interm_data_names,
            "repul_w": cfg.behaviours.class_m.model.repul_w,
            "min_dist": cfg.behaviours.class_m.model.min_dist,
            "max_dist": cfg.behaviours.class_m.model.max_dist,
            "max_long_vel": cfg.behaviours.class_m.max_long_vel,
            "max_rot_vel": cfg.behaviours.class_m.max_rot_vel,
            "max_abs_f": cfg.behaviours.class_m.max_abs_f,
        },
    )
    assert (
        len(out_step1) == 2
    ), "Should return next swarm configuration and the interm_data dict."
    assert isinstance(
        out_step1[0], Boids_BallisticRW
    ), "Should return a Boids_BallisticRW namedtuple."
    assert isinstance(out_step1[1], dict), "Should return a interm_data dict."

    cfg.behaviours.class_m.storing_data = False
    out_step2 = ballistic_random_walk_step(
        out_step1[0],
        {
            "box_init": cfg.behaviours.class_m.box_init,
            "box_size": cfg.behaviours.class_m.box_size,
            "dt": cfg.behaviours.class_m.dt,
            "inner_loop": cfg.behaviours.class_m.inner_loop,
            "sim_speed": cfg.behaviours.class_m.sim_speed,
            "f_lim": cfg.behaviours.class_m.max_abs_f,
            "v_lim": cfg.behaviours.class_m.max_abs_sp,
            "max_speed": cfg.behaviours.class_m.max_abs_sp,
            "v_const": cfg.behaviours.class_m.init_speed,
            "eps": cfg.behaviours.class_m.eps,
            "walls_b": cfg.behaviours.class_m.walls_b,
            "store_interm_data": cfg.behaviours.class_m.storing_data,
            "interm_data_names": cfg.behaviours.class_m.model.interm_data_names,
            "repul_w": cfg.behaviours.class_m.model.repul_w,
            "min_dist": cfg.behaviours.class_m.model.min_dist,
            "max_dist": cfg.behaviours.class_m.model.max_dist,
            "max_long_vel": cfg.behaviours.class_m.max_long_vel,
            "max_rot_vel": cfg.behaviours.class_m.max_rot_vel,
            "max_abs_f": cfg.behaviours.class_m.max_abs_f,
        },
    )
    assert len(out_step2) == 2, "Should only return next swarm configuration."
    assert isinstance(
        out_step2, Boids_BallisticRW
    ), "Should return a Boids_BallisticRW namedtuple."
    assert not np.all(
        out_step2.X == out_step1[0].X, axis=1
    ).all(), "Should be different swarm configurations."

    # Exporting the frames to a video for visual inspection
    render_boids_gif(
        cfg.behaviours.class_m.box_size,
        [out_step2],
        output_dir + "/unit_t2/frames.gif",
        cfg.behaviours.agent_trace_steps,
        cfg.behaviours.fps_render,
        cfg.behaviours.class_m.model.max_dist,
        cfg.behaviours.class_m.init_speed,
    )
    render_network_gif(
        cfg.behaviours.class_m.box_size,
        [out_step2],
        output_dir + "/unit_t2/graph_frames.gif",
        cfg.behaviours.fps_render,
        cfg.behaviours.class_m.model.max_dist,
        cfg.behaviours.class_m.walls_b,
    )


def unit_t3(output_dir: str, state: dict, cfg: DictConfig) -> None:
    """
    Simulating with the Ballistic Motion model
    """
    boids_buffer = [state["boids"]]
    os.makedirs(output_dir + f"/unit_t3/", mode=0o777, exist_ok=True)

    # Simulation loop
    boids_rand_walk = boids_buffer[0]
    for _ in range(cfg.behaviours.class_m.steps):
        out_step = ballistic_random_walk_step(
            boids_rand_walk,
            {
                "box_init": cfg.behaviours.class_m.box_init,
                "box_size": cfg.behaviours.class_m.box_size,
                "dt": cfg.behaviours.class_m.dt,
                "inner_loop": cfg.behaviours.class_m.inner_loop,
                "sim_speed": cfg.behaviours.class_m.sim_speed,
                "f_lim": cfg.behaviours.class_m.max_abs_f,
                "v_lim": cfg.behaviours.class_m.max_abs_sp,
                "max_speed": cfg.behaviours.class_m.max_abs_sp,
                "v_const": cfg.behaviours.class_m.init_speed,
                "eps": cfg.behaviours.class_m.eps,
                "walls_b": cfg.behaviours.class_m.walls_b,
                "store_interm_data": cfg.behaviours.class_m.storing_data,
                "interm_data_names": cfg.behaviours.class_m.model.interm_data_names,
                "repul_w": cfg.behaviours.class_m.model.repul_w,
                "min_dist": cfg.behaviours.class_m.model.min_dist,
                "max_dist": cfg.behaviours.class_m.model.max_dist,
                "max_long_vel": cfg.behaviours.class_m.max_long_vel,
                "max_rot_vel": cfg.behaviours.class_m.max_rot_vel,
                "max_abs_f": cfg.behaviours.class_m.max_abs_f,
            },
        )

        if cfg.behaviours.class_m.storing_data:
            boids_rand_walk, _ = out_step
        else:
            boids_rand_walk = out_step

        boids_buffer.append(boids_rand_walk)

        # Checking that all agents' speed is constant
        assert np.any(
            np.abs(
                np.linalg.norm(boids_rand_walk.X_dot, axis=1)
                - cfg.behaviours.class_m.init_speed
            )
            < 1e-1
        ), "All agent's speeds should be constant."

    # Visualization for random walk forces
    for idx in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        repulsive_f_f, _, _ = ballistic_random_walk_force(
            boids_buffer[-idx].X,
            {
                "box_size": cfg.behaviours.class_m.box_size,
                "walls_b": cfg.behaviours.class_m.walls_b,
                "min_dist": cfg.behaviours.class_m.model.min_dist,
                "max_dist": cfg.behaviours.class_m.model.max_dist,
                "max_long_vel": cfg.behaviours.class_m.max_long_vel,
                "max_rot_vel": cfg.behaviours.class_m.max_rot_vel,
                "max_abs_f": cfg.behaviours.class_m.max_abs_f,
            },
        )
        repulsive_f_f_x, repulsive_f_f_y = np.split(repulsive_f_f, 2, axis=1)
        visualizing_boids_force(
            boids_buffer[-1].X,
            (
                (
                    repulsive_f_f_x,
                    repulsive_f_f_y,
                    "red",
                    cfg.behaviours.class_m.box_size // 3,
                    "Repulsion force",
                ),
            ),
            output_dir + f"/unit_t3/boids_forces_{-idx}.png",
            "Ballistic Motion Boid Forces",
        )
    # Exporting the frames to a video for visual inspection
    render_boids_gif(
        cfg.behaviours.class_m.box_size,
        boids_buffer[-(len(boids_buffer) // 20) :],
        output_dir + "/unit_t3/frames.gif",
        cfg.behaviours.agent_trace_steps,
        cfg.behaviours.fps_render,
        cfg.behaviours.class_m.model.max_dist,
        cfg.behaviours.class_m.init_speed,
    )
    # Exporting the frames for visual inspection
    render_boids_sakana(
        deepcopy(boids_buffer),
        {
            "space_size": 1.0,
            "img_size": int(cfg.behaviours.class_m.box_size),
            "bird_render_size": 0.007,  # 0.0015,  # 0.015,
            "bird_render_sharpness": 40.0,
            "red_boids": [],
            "v_const": cfg.behaviours.class_m.init_speed,
            "out_name": output_dir + "/unit_t3/sakana_frames.png",
            "fps": cfg.behaviours.fps_render,
        },
        time_sampling=(8, True),
    )


def unit_t4(output_dir: str, cfg: DictConfig) -> None:
    """
    Testing boids initialization, brownian motion forces
        computation and forces rendering basics.
    """
    os.makedirs(output_dir + f"/unit_t4/", mode=0o777, exist_ok=True)

    # Test initialization
    boids_brownian = initialize_boids_random_walk(
        cfg.behaviours.class_m.box_size,
        cfg.behaviours.class_m.boid_count,
        cfg.behaviours.class_m.dim,
        cfg.behaviours.class_m.seed,
        cfg.behaviours.class_m.init_speed,
        cfg.behaviours.class_m.model.name,
        (cfg.behaviours.class_m.model.mu, cfg.behaviours.class_m.model.beta_scale),
    )
    assert boids_brownian[0].shape == (
        cfg.behaviours.class_m.boid_count,
        cfg.behaviours.class_m.dim,
    ), "Should be the same shape."
    assert (
        boids_brownian[1].shape == boids_brownian[0].shape
    ), "Should be the same shape."
    assert (
        boids_brownian[2].shape != boids_brownian[1].shape
    ), "Should be the same shape."
    assert np.isclose(
        np.mean(np.linalg.norm(boids_brownian[1], axis=-1)),
        cfg.behaviours.class_m.init_speed,
        1e-3,
    ), "The inital speed must be constant."

    # Test random walk forces computation
    total_force1, _, _ = brownian_random_walk_force(
        boids_brownian.X,
        boids_brownian.Counter,
        {
            "box_size": cfg.behaviours.class_m.box_size,
            "walls_b": cfg.behaviours.class_m.walls_b,
            "min_dist": cfg.behaviours.class_m.model.min_dist,
            "max_dist": cfg.behaviours.class_m.model.max_dist,
            "max_long_vel": cfg.behaviours.class_m.max_long_vel,
            "max_rot_vel": cfg.behaviours.class_m.max_rot_vel,
            "max_abs_f": cfg.behaviours.class_m.max_abs_f,
            "repul_w": cfg.behaviours.class_m.model.repul_w,
            "rand_w": cfg.behaviours.class_m.model.rand_w,
            "rho": cfg.behaviours.class_m.model.rho,
        },
    )
    assert total_force1.shape == (
        cfg.behaviours.class_m.boid_count,
        cfg.behaviours.class_m.dim,
    ), "Should be the same shape."

    # Visualization for random walk forces
    total_force1_x, total_force1_y = np.split(total_force1, 2, axis=1)
    visualizing_boids_force(
        boids_brownian.X,
        (
            (
                total_force1_x,
                total_force1_y,
                "red",
                cfg.behaviours.class_m.box_size // 3,
                "Total force",
            ),
        ),
        output_dir + f"/unit_t4/boids_forces.png",
        "Brownian Motion Boid Forces",
    )


def unit_t5(output_dir: str, state: dict, cfg: DictConfig) -> None:
    """
    Using the Brownian Motion model for a single step
    """
    os.makedirs(output_dir + f"/unit_t5/", mode=0o777, exist_ok=True)

    cfg.behaviours.class_m.storing_data = True
    out_step1 = brownian_random_walk_step(
        state["boids"],
        {
            "box_init": cfg.behaviours.class_m.box_init,
            "box_size": cfg.behaviours.class_m.box_size,
            "dt": cfg.behaviours.class_m.dt,
            "inner_loop": cfg.behaviours.class_m.inner_loop,
            "sim_speed": cfg.behaviours.class_m.sim_speed,
            "f_lim": cfg.behaviours.class_m.max_abs_f,
            "v_lim": cfg.behaviours.class_m.max_abs_sp,
            "max_speed": cfg.behaviours.class_m.max_abs_sp,
            "v_const": cfg.behaviours.class_m.init_speed,
            "eps": cfg.behaviours.class_m.eps,
            "walls_b": cfg.behaviours.class_m.walls_b,
            "store_interm_data": cfg.behaviours.class_m.storing_data,
            "interm_data_names": cfg.behaviours.class_m.model.interm_data_names,
            "repul_w": cfg.behaviours.class_m.model.repul_w,
            "rand_w": cfg.behaviours.class_m.model.rand_w,
            "min_dist": cfg.behaviours.class_m.model.min_dist,
            "max_dist": cfg.behaviours.class_m.model.max_dist,
            "max_long_vel": cfg.behaviours.class_m.max_long_vel,
            "max_rot_vel": cfg.behaviours.class_m.max_rot_vel,
            "max_abs_f": cfg.behaviours.class_m.max_abs_f,
            "mu": cfg.behaviours.class_m.model.mu,
            "beta_scale": cfg.behaviours.class_m.model.beta_scale,
            "rho": cfg.behaviours.class_m.model.rho,
        },
    )
    assert (
        len(out_step1) == 2
    ), "Should return next swarm configuration and the interm_data dict."
    assert isinstance(
        out_step1[0], Boids_BrownianRW
    ), "Should return a Boids_BrownianRW namedtuple."
    assert isinstance(out_step1[1], dict), "Should return a interm_data dict."

    cfg.behaviours.class_m.storing_data = False
    out_step2 = brownian_random_walk_step(
        out_step1[0],
        {
            "box_init": cfg.behaviours.class_m.box_init,
            "box_size": cfg.behaviours.class_m.box_size,
            "dt": cfg.behaviours.class_m.dt,
            "inner_loop": cfg.behaviours.class_m.inner_loop,
            "sim_speed": cfg.behaviours.class_m.sim_speed,
            "f_lim": cfg.behaviours.class_m.max_abs_f,
            "v_lim": cfg.behaviours.class_m.max_abs_sp,
            "max_speed": cfg.behaviours.class_m.max_abs_sp,
            "v_const": cfg.behaviours.class_m.init_speed,
            "eps": cfg.behaviours.class_m.eps,
            "walls_b": cfg.behaviours.class_m.walls_b,
            "store_interm_data": cfg.behaviours.class_m.storing_data,
            "interm_data_names": cfg.behaviours.class_m.model.interm_data_names,
            "repul_w": cfg.behaviours.class_m.model.repul_w,
            "rand_w": cfg.behaviours.class_m.model.rand_w,
            "min_dist": cfg.behaviours.class_m.model.min_dist,
            "max_dist": cfg.behaviours.class_m.model.max_dist,
            "max_long_vel": cfg.behaviours.class_m.max_long_vel,
            "max_rot_vel": cfg.behaviours.class_m.max_rot_vel,
            "max_abs_f": cfg.behaviours.class_m.max_abs_f,
            "mu": cfg.behaviours.class_m.model.mu,
            "beta_scale": cfg.behaviours.class_m.model.beta_scale,
            "rho": cfg.behaviours.class_m.model.rho,
        },
    )
    assert len(out_step2) == 3, "Should only return next swarm configuration."
    assert isinstance(
        out_step2, Boids_BrownianRW
    ), "Should return a Boids_BrownianRW namedtuple."
    assert not np.all(
        out_step2.X == out_step1[0].X, axis=1
    ).all(), "Should be different swarm configurations."

    # Exporting the frames to a video for visual inspection
    render_boids_gif(
        cfg.behaviours.class_m.box_size,
        [out_step2],
        output_dir + "/unit_t5/frames.gif",
        cfg.behaviours.agent_trace_steps,
        cfg.behaviours.fps_render,
        cfg.behaviours.class_m.model.max_dist,
        cfg.behaviours.class_m.init_speed,
    )
    render_network_gif(
        cfg.behaviours.class_m.box_size,
        [out_step2],
        output_dir + "/unit_t5/graph_frames.gif",
        cfg.behaviours.fps_render,
        cfg.behaviours.class_m.model.max_dist,
        cfg.behaviours.class_m.walls_b,
    )


def unit_t6(output_dir: str, state: dict, cfg: DictConfig) -> None:
    """
    Simulating with the Brownian Motion model
    """
    boids_buffer = [state["boids"]]
    os.makedirs(output_dir + f"/unit_t6/", mode=0o777, exist_ok=True)

    # Simulation loop
    boids_rand_walk = boids_buffer[0]
    for _ in range(cfg.behaviours.class_m.steps):
        out_step = brownian_random_walk_step(
            boids_rand_walk,
            {
                "box_init": cfg.behaviours.class_m.box_init,
                "box_size": cfg.behaviours.class_m.box_size,
                "dt": cfg.behaviours.class_m.dt,
                "inner_loop": cfg.behaviours.class_m.inner_loop,
                "sim_speed": cfg.behaviours.class_m.sim_speed,
                "f_lim": cfg.behaviours.class_m.max_abs_f,
                "v_lim": cfg.behaviours.class_m.max_abs_sp,
                "max_speed": cfg.behaviours.class_m.max_abs_sp,
                "v_const": cfg.behaviours.class_m.init_speed,
                "eps": cfg.behaviours.class_m.eps,
                "walls_b": cfg.behaviours.class_m.walls_b,
                "store_interm_data": cfg.behaviours.class_m.storing_data,
                "interm_data_names": cfg.behaviours.class_m.model.interm_data_names,
                "repul_w": cfg.behaviours.class_m.model.repul_w,
                "rand_w": cfg.behaviours.class_m.model.rand_w,
                "min_dist": cfg.behaviours.class_m.model.min_dist,
                "max_dist": cfg.behaviours.class_m.model.max_dist,
                "max_long_vel": cfg.behaviours.class_m.max_long_vel,
                "max_rot_vel": cfg.behaviours.class_m.max_rot_vel,
                "max_abs_f": cfg.behaviours.class_m.max_abs_f,
                "mu": cfg.behaviours.class_m.model.mu,
                "beta_scale": cfg.behaviours.class_m.model.beta_scale,
                "rho": cfg.behaviours.class_m.model.rho,
            },
        )

        if cfg.behaviours.class_m.storing_data:
            boids_rand_walk, _ = out_step
        else:
            boids_rand_walk = out_step

        boids_buffer.append(boids_rand_walk)

    # Visualization for random walk forces
    for idx in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        repulsive_f_f, _, _ = brownian_random_walk_force(
            boids_buffer[-idx].X,
            boids_buffer[-idx].Counter,
            {
                "box_size": cfg.behaviours.class_m.box_size,
                "walls_b": cfg.behaviours.class_m.walls_b,
                "min_dist": cfg.behaviours.class_m.model.min_dist,
                "max_dist": cfg.behaviours.class_m.model.max_dist,
                "max_long_vel": cfg.behaviours.class_m.max_long_vel,
                "max_rot_vel": cfg.behaviours.class_m.max_rot_vel,
                "max_abs_f": cfg.behaviours.class_m.max_abs_f,
                "repul_w": cfg.behaviours.class_m.model.repul_w,
                "rand_w": cfg.behaviours.class_m.model.rand_w,
                "rho": cfg.behaviours.class_m.model.rho,
            },
        )
        repulsive_f_f_x, repulsive_f_f_y = np.split(repulsive_f_f, 2, axis=1)
        visualizing_boids_force(
            boids_buffer[-1].X,
            (
                (
                    repulsive_f_f_x,
                    repulsive_f_f_y,
                    "red",
                    cfg.behaviours.class_m.box_size // 3,
                    "Total force",
                ),
            ),
            output_dir + f"/unit_t6/boids_forces_{-idx}.png",
            "Brownian Motion Boid Forces",
        )
    # Exporting the frames to a video for visual inspection
    render_boids_gif(
        cfg.behaviours.class_m.box_size,
        boids_buffer[-(len(boids_buffer) // 20) :],
        output_dir + "/unit_t6/frames.gif",
        cfg.behaviours.agent_trace_steps,
        cfg.behaviours.fps_render,
        cfg.behaviours.class_m.model.max_dist,
        cfg.behaviours.class_m.init_speed,
    )
    # Exporting the frames for visual inspection
    render_boids_sakana(
        deepcopy(boids_buffer),
        {
            "space_size": 1.0,
            "img_size": int(cfg.behaviours.class_m.box_size),
            "bird_render_size": 0.007,  # 0.0015,  # 0.015,
            "bird_render_sharpness": 40.0,
            "red_boids": [],
            "v_const": cfg.behaviours.class_m.init_speed,
            "out_name": output_dir + "/unit_t6/sakana_frames.png",
            "fps": cfg.behaviours.fps_render,
        },
        time_sampling=(8, True),
    )


@hydra.main(config_path="../swarm_hydra/configs", config_name="config", version_base="1.1")
def main(cfg: DictConfig):
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    hydra_cfg["runtime"]["output_dir"]
    out_dir = hydra_cfg["runtime"]["output_dir"]

    # Redirect stdout & stderr
    sys.stdout = LoggerWriter(log, logging.INFO)
    sys.stderr = LoggerWriter(log, logging.ERROR)

    reset_seeds(cfg.behaviours.class_m.seed)

    cfg.behaviours.class_m = cfg.behaviours.random_walk

    # Ballistic Random Walk tests
    cfg.behaviours.class_m.model = cfg.behaviours.random_walk.ballistic
    unit_t1(out_dir, cfg)
    boids_ballistic = initialize_boids_random_walk(
        cfg.behaviours.class_m.box_size,
        cfg.behaviours.class_m.boid_count,
        cfg.behaviours.class_m.dim,
        cfg.behaviours.class_m.seed,
        cfg.behaviours.class_m.init_speed,
        cfg.behaviours.class_m.model.name,
        (None, None),
    )
    state2 = {"boids": deepcopy(boids_ballistic)}
    unit_t2(out_dir, state2, cfg)
    state3 = {"boids": deepcopy(boids_ballistic)}
    unit_t3(out_dir, state3, cfg)

    # Brownian Motion Random Walk tests
    cfg.behaviours.class_m.model = cfg.behaviours.random_walk.brownian
    unit_t4(out_dir, cfg)
    boids_brownian = initialize_boids_random_walk(
        cfg.behaviours.class_m.box_size,
        cfg.behaviours.class_m.boid_count,
        cfg.behaviours.class_m.dim,
        cfg.behaviours.class_m.seed,
        cfg.behaviours.class_m.init_speed,
        cfg.behaviours.class_m.model.name,
        (cfg.behaviours.class_m.model.mu, cfg.behaviours.class_m.model.beta_scale),
    )
    state2 = {"boids": deepcopy(boids_brownian)}
    unit_t5(out_dir, state2, cfg)
    state3 = {"boids": deepcopy(boids_brownian)}
    unit_t6(out_dir, state3, cfg)

    # Exporting the updated config
    with open(out_dir + "/.hydra/runtime_config_reynolds.yaml", "w") as f:
        OmegaConf.save(config=cfg, f=f.name)


if __name__ == "__main__":
    main()
