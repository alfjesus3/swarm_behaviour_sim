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
from swarm_hydra.behaviours.dispersion import *


# A logger for this file
log = logging.getLogger(__name__)


def unit_t1(output_dir: str, cfg: DictConfig) -> None:
    """
    Testing boids initialization, dispersion forces
        computation and potential field rendering basics.
    """
    os.makedirs(output_dir + f"/unit_t1/", mode=0o777, exist_ok=True)

    # Test initialization
    boids_disper = initialize_boids_disperation(
        cfg.behaviours.class_m.box_size,
        cfg.behaviours.class_m.boid_count,
        cfg.behaviours.class_m.dim,
        cfg.behaviours.class_m.seed,
        cfg.behaviours.class_m.init_speed,
    )
    assert boids_disper[0].shape == (
        cfg.behaviours.class_m.boid_count,
        cfg.behaviours.class_m.dim,
    ), "Should be the same shape."
    assert boids_disper[1].shape == boids_disper[0].shape, "Should be the same shape."
    assert np.isclose(
        np.mean(np.linalg.norm(boids_disper[1], axis=-1)),
        cfg.behaviours.class_m.init_speed,
        1e-3,
    ), "The inital speed must be constant."

    # Test repulsive potential field forces computation
    repulsive_f1, _, _ = repulsive_field_dispersion_force(
        boids_disper.X,
        {
            "box_size": cfg.behaviours.class_m.box_size,
            "walls_b": cfg.behaviours.class_m.walls_b,
            "min_dist": cfg.behaviours.class_m.model.min_dist,
            "max_dist": cfg.behaviours.class_m.model.max_dist,
            "max_long_vel": cfg.behaviours.class_m.max_long_vel,
            "max_rot_vel": cfg.behaviours.class_m.max_rot_vel,
        },
    )
    assert repulsive_f1.shape == (
        cfg.behaviours.class_m.boid_count,
        cfg.behaviours.class_m.dim,
    ), "Should be the same shape."

    # Visualization for repulsive potential field forces
    repulsive_f1_x, repulsive_f1_y = np.split(repulsive_f1, 2, axis=1)
    visualizing_boids_force(
        boids_disper.X,
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
        "Dispersion Boids Forces",
    )


def unit_t2(output_dir: str, state: dict, cfg: DictConfig) -> None:
    """
    Using the Dispersion model for a single step
    """
    os.makedirs(output_dir + f"/unit_t2/", mode=0o777, exist_ok=True)

    cfg.behaviours.class_m.storing_data = True
    out_step1 = repulsive_field_dispersion_step(
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
        },
    )
    assert (
        len(out_step1) == 2
    ), "Should return next swarm configuration and the interm_data dict."
    assert isinstance(
        out_step1[0], Boids_Dispersion
    ), "Should return a Boids_Dispersion namedtuple."
    assert isinstance(out_step1[1], dict), "Should return a interm_data dict."

    cfg.behaviours.class_m.storing_data = False
    out_step2 = repulsive_field_dispersion_step(
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
        },
    )
    assert len(out_step2) == 2, "Should only return next swarm configuration."
    assert isinstance(
        out_step2, Boids_Dispersion
    ), "Should return a Boids_Dispersion namedtuple."
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
    Using the 'sort-of' alignment force Sicsek flocking model
    """
    boids_buffer = [state["boids"]]
    os.makedirs(output_dir + f"/unit_t3/", mode=0o777, exist_ok=True)

    # Simulation loop
    boids_disper = boids_buffer[0]
    for _ in range(cfg.behaviours.class_m.steps):
        out_step = repulsive_field_dispersion_step(
            boids_disper,
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
            },
        )

        if cfg.behaviours.class_m.storing_data:
            boids_disper, _ = out_step
        else:
            boids_disper = out_step

        boids_buffer.append(boids_disper)

        # Checking that all agents' speed is constant
        assert np.any(
            np.abs(
                np.linalg.norm(boids_disper.X_dot, axis=1)
                - cfg.behaviours.class_m.init_speed
            )
            < 1e-1
        ), "All agent's speeds should be constant."

    # Visualization for repulsive potential field forces
    for idx in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        repulsive_f_f, _, _ = repulsive_field_dispersion_force(
            boids_buffer[-idx].X,
            {
                "box_size": cfg.behaviours.class_m.box_size,
                "walls_b": cfg.behaviours.class_m.walls_b,
                "min_dist": cfg.behaviours.class_m.model.min_dist,
                "max_dist": cfg.behaviours.class_m.model.max_dist,
                "max_long_vel": cfg.behaviours.class_m.max_long_vel,
                "max_rot_vel": cfg.behaviours.class_m.max_rot_vel,
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
            "Dispersion Boids Forces",
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


@hydra.main(config_path="../swarm_hydra/configs", config_name="config", version_base="1.1")
def main(cfg: DictConfig):
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    hydra_cfg["runtime"]["output_dir"]
    out_dir = hydra_cfg["runtime"]["output_dir"]

    # Redirect stdout & stderr
    sys.stdout = LoggerWriter(log, logging.INFO)
    sys.stderr = LoggerWriter(log, logging.ERROR)

    reset_seeds(cfg.behaviours.class_m.seed)

    # Repulsive Potential Field Dispersion tests
    cfg.behaviours.class_m = cfg.behaviours.dispersion
    cfg.behaviours.class_m.model = cfg.behaviours.dispersion.repu_field
    unit_t1(out_dir, cfg)
    boids_disper = initialize_boids_disperation(
        cfg.behaviours.class_m.box_size,
        cfg.behaviours.class_m.boid_count,
        cfg.behaviours.class_m.dim,
        cfg.behaviours.class_m.seed,
        cfg.behaviours.class_m.init_speed,
    )
    state2 = {"boids": deepcopy(boids_disper)}
    unit_t2(out_dir, state2, cfg)
    state3 = {"boids": deepcopy(boids_disper)}
    unit_t3(out_dir, state3, cfg)

    # Exporting the updated config
    with open(out_dir + "/.hydra/runtime_config_reynolds.yaml", "w") as f:
        OmegaConf.save(config=cfg, f=f.name)


if __name__ == "__main__":
    main()
