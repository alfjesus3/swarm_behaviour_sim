import hydra
from omegaconf import DictConfig, OmegaConf
import logging

import numpy as np
import os
import sys
from copy import deepcopy

from swarm_hydra.entry_point import *
from swarm_hydra.behaviours.utils_behaviours import *
from swarm_hydra.behaviours.flocking import *


# A logger for this file
log = logging.getLogger(__name__)


def unit_t1(output_dir: str, state: dict, cfg: DictConfig) -> None:
    """
    Reynolds flocking model w/ the alignment force
    """
    boids_buffer = [state["boids"]]
    os.makedirs(output_dir + f"/unit_t1/", mode=0o777, exist_ok=True)

    # Simulation loop
    boids_rey = boids_buffer[0]
    for _ in range(cfg.behaviours.class_m.steps):
        out_step = original_reynolds_step(
            boids_rey,
            {
                "box_init": cfg.behaviours.class_m.box_init,
                "max_dist": cfg.behaviours.class_m.model.alignm_dist_thres,
                "box_size": cfg.behaviours.class_m.box_size,
                "dt": cfg.behaviours.class_m.dt,
                "inner_loop": cfg.behaviours.class_m.inner_loop,
                "sim_speed": cfg.behaviours.class_m.sim_speed,
                "max_speed": cfg.behaviours.class_m.max_abs_sp,
                "coef_alignment": cfg.behaviours.class_m.model.alignm_w,
                "d_align": cfg.behaviours.class_m.model.alignm_dist_thres,
                "coef_cohesion": 0.0,
                "d_cohe": cfg.behaviours.class_m.model.cohe_dist_thres,
                "coef_separation": 0.0,
                "d_sepa": cfg.behaviours.class_m.model.separ_dist_thres,
                "f_lim": cfg.behaviours.class_m.max_abs_f,
                "v_lim": cfg.behaviours.class_m.max_abs_sp,
                "v_const": cfg.behaviours.class_m.init_speed,
                "eps": cfg.behaviours.class_m.eps,
                "walls_b": cfg.behaviours.class_m.walls_b,
                "store_interm_data": cfg.behaviours.class_m.storing_data,
                "interm_data_names": cfg.behaviours.class_m.model.interm_data_names,
            },
        )

        if cfg.behaviours.class_m.storing_data:
            boids_rey, _ = out_step
        else:
            boids_rey = out_step

        boids_buffer.append(boids_rey)

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
            "out_name": output_dir + "/unit_t1/sakana_frames.png",
            "fps": cfg.behaviours.fps_render,
        },
        time_sampling=(8, True),
    )


def unit_t2(output_dir: str, state: dict, cfg: DictConfig) -> None:
    """
    Reynolds flocking model w/ the cohesion force
    """
    boids_buffer = [state["boids"]]
    os.makedirs(output_dir + f"/unit_t2/", mode=0o777, exist_ok=True)

    # Simulation loop
    boids_rey = boids_buffer[0]
    for _ in range(cfg.behaviours.class_m.steps):
        out_step = original_reynolds_step(
            boids_rey,
            {
                "box_init": cfg.behaviours.class_m.box_init,
                "max_dist": cfg.behaviours.class_m.model.cohe_dist_thres,
                "box_size": cfg.behaviours.class_m.box_size,
                "dt": cfg.behaviours.class_m.dt,
                "inner_loop": cfg.behaviours.class_m.inner_loop,
                "sim_speed": cfg.behaviours.class_m.sim_speed,
                "max_speed": cfg.behaviours.class_m.max_abs_sp,
                "coef_alignment": 0.0,
                "d_align": cfg.behaviours.class_m.model.alignm_dist_thres,
                "coef_cohesion": cfg.behaviours.class_m.model.cohe_w,
                "d_cohe": cfg.behaviours.class_m.model.cohe_dist_thres,
                "coef_separation": 0.0,
                "d_sepa": cfg.behaviours.class_m.model.separ_dist_thres,
                "f_lim": cfg.behaviours.class_m.max_abs_f,
                "v_lim": cfg.behaviours.class_m.max_abs_sp,
                "v_const": cfg.behaviours.class_m.init_speed,
                "eps": cfg.behaviours.class_m.eps,
                "walls_b": cfg.behaviours.class_m.walls_b,
                "store_interm_data": cfg.behaviours.class_m.storing_data,
                "interm_data_names": cfg.behaviours.class_m.model.interm_data_names,
            },
        )

        if cfg.behaviours.class_m.storing_data:
            boids_rey, _ = out_step
        else:
            boids_rey = out_step

        boids_buffer.append(boids_rey)

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
            "out_name": output_dir + "/unit_t2/sakana_frames.png",
            "fps": cfg.behaviours.fps_render,
        },
        time_sampling=(8, True),
    )


def unit_t3(output_dir: str, state: dict, cfg: DictConfig) -> None:
    """
    Reynolds flocking model w/ the separation force
    """
    boids_buffer = [state["boids"]]
    os.makedirs(output_dir + f"/unit_t3/", mode=0o777, exist_ok=True)

    # Simulation loop
    boids_rey = boids_buffer[0]
    for _ in range(cfg.behaviours.class_m.steps):
        out_step = original_reynolds_step(
            boids_rey,
            {
                "box_init": cfg.behaviours.class_m.box_init,
                "max_dist": cfg.behaviours.class_m.model.separ_dist_thres,
                "box_size": cfg.behaviours.class_m.box_size,
                "dt": cfg.behaviours.class_m.dt,
                "inner_loop": cfg.behaviours.class_m.inner_loop,
                "sim_speed": cfg.behaviours.class_m.sim_speed,
                "max_speed": cfg.behaviours.class_m.max_abs_sp,
                "coef_alignment": 0.0,
                "d_align": cfg.behaviours.class_m.model.alignm_dist_thres,
                "coef_cohesion": 0.0,
                "d_cohe": cfg.behaviours.class_m.model.cohe_dist_thres,
                "coef_separation": cfg.behaviours.class_m.model.separ_w,
                "d_sepa": cfg.behaviours.class_m.model.separ_dist_thres,
                "f_lim": cfg.behaviours.class_m.max_abs_f,
                "v_lim": cfg.behaviours.class_m.max_abs_sp,
                "v_const": cfg.behaviours.class_m.init_speed,
                "eps": cfg.behaviours.class_m.eps,
                "walls_b": cfg.behaviours.class_m.walls_b,
                "store_interm_data": cfg.behaviours.class_m.storing_data,
                "interm_data_names": cfg.behaviours.class_m.model.interm_data_names,
            },
        )

        if cfg.behaviours.class_m.storing_data:
            boids_rey, _ = out_step
        else:
            boids_rey = out_step

        boids_buffer.append(boids_rey)

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


def unit_t4(output_dir: str, state: dict, cfg: DictConfig) -> None:
    """
    Using the alignment + cohesion forces Reynolds flocking model
    """
    boids_buffer = [state["boids"]]
    os.makedirs(output_dir + f"/unit_t4/", mode=0o777, exist_ok=True)

    # Simulation loop
    boids_rey = boids_buffer[0]
    for _ in range(cfg.behaviours.class_m.steps):
        out_step = original_reynolds_step(
            boids_rey,
            {
                "box_init": cfg.behaviours.class_m.box_init,
                "max_dist": cfg.behaviours.class_m.model.alignm_dist_thres,
                "box_size": cfg.behaviours.class_m.box_size,
                "dt": cfg.behaviours.class_m.dt,
                "inner_loop": cfg.behaviours.class_m.inner_loop,
                "sim_speed": cfg.behaviours.class_m.sim_speed,
                "max_speed": cfg.behaviours.class_m.max_abs_sp,
                "coef_alignment": cfg.behaviours.class_m.model.alignm_w,
                "d_align": cfg.behaviours.class_m.model.alignm_dist_thres,
                "coef_cohesion": cfg.behaviours.class_m.model.cohe_w,
                "d_cohe": cfg.behaviours.class_m.model.cohe_dist_thres,
                "coef_separation": 0.0,
                "d_sepa": cfg.behaviours.class_m.model.separ_dist_thres,
                "f_lim": cfg.behaviours.class_m.max_abs_f,
                "v_lim": cfg.behaviours.class_m.max_abs_sp,
                "v_const": cfg.behaviours.class_m.init_speed,
                "eps": cfg.behaviours.class_m.eps,
                "walls_b": cfg.behaviours.class_m.walls_b,
                "store_interm_data": cfg.behaviours.class_m.storing_data,
                "interm_data_names": cfg.behaviours.class_m.model.interm_data_names,
            },
        )

        if cfg.behaviours.class_m.storing_data:
            boids_rey, _ = out_step
        else:
            boids_rey = out_step

        boids_buffer.append(boids_rey)

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
            "out_name": output_dir + "/unit_t4/sakana_frames.png",
            "fps": cfg.behaviours.fps_render,
        },
        time_sampling=(8, True),
    )


def unit_t5(output_dir: str, state: dict, cfg: DictConfig) -> None:
    """
    Using the alignment + separation forces Reynolds flocking model
    """
    boids_buffer = [state["boids"]]
    os.makedirs(output_dir + f"/unit_t5/", mode=0o777, exist_ok=True)

    # Simulation loop
    boids_rey = boids_buffer[0]
    for _ in range(cfg.behaviours.class_m.steps):
        out_step = original_reynolds_step(
            boids_rey,
            {
                "box_init": cfg.behaviours.class_m.box_init,
                "max_dist": cfg.behaviours.class_m.model.alignm_dist_thres,
                "box_size": cfg.behaviours.class_m.box_size,
                "dt": cfg.behaviours.class_m.dt,
                "inner_loop": cfg.behaviours.class_m.inner_loop,
                "sim_speed": cfg.behaviours.class_m.sim_speed,
                "max_speed": cfg.behaviours.class_m.max_abs_sp,
                "coef_alignment": cfg.behaviours.class_m.model.alignm_w,
                "d_align": cfg.behaviours.class_m.model.alignm_dist_thres,
                "coef_cohesion": 0.0,
                "d_cohe": cfg.behaviours.class_m.model.cohe_dist_thres,
                "coef_separation": cfg.behaviours.class_m.model.separ_w,
                "d_sepa": cfg.behaviours.class_m.model.separ_dist_thres,
                "f_lim": cfg.behaviours.class_m.max_abs_f,
                "v_lim": cfg.behaviours.class_m.max_abs_sp,
                "v_const": cfg.behaviours.class_m.init_speed,
                "eps": cfg.behaviours.class_m.eps,
                "walls_b": cfg.behaviours.class_m.walls_b,
                "store_interm_data": cfg.behaviours.class_m.storing_data,
                "interm_data_names": cfg.behaviours.class_m.model.interm_data_names,
            },
        )

        if cfg.behaviours.class_m.storing_data:
            boids_rey, _ = out_step
        else:
            boids_rey = out_step

        boids_buffer.append(boids_rey)

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
            "out_name": output_dir + "/unit_t5/sakana_frames.png",
            "fps": cfg.behaviours.fps_render,
        },
        time_sampling=(8, True),
    )


def unit_t6(output_dir: str, state: dict, cfg: DictConfig) -> None:
    """
    Using the cohesion + separation forces Reynolds flocking model
    """
    boids_buffer = [state["boids"]]
    os.makedirs(output_dir + f"/unit_t6/", mode=0o777, exist_ok=True)

    # Simulation loop
    boids_rey = boids_buffer[0]
    for _ in range(cfg.behaviours.class_m.steps):
        out_step = original_reynolds_step(
            boids_rey,
            {
                "box_init": cfg.behaviours.class_m.box_init,
                "max_dist": cfg.behaviours.class_m.model.cohe_dist_thres,
                "box_size": cfg.behaviours.class_m.box_size,
                "dt": cfg.behaviours.class_m.dt,
                "inner_loop": cfg.behaviours.class_m.inner_loop,
                "sim_speed": cfg.behaviours.class_m.sim_speed,
                "max_speed": cfg.behaviours.class_m.max_abs_sp,
                "coef_alignment": 0.0,
                "d_align": cfg.behaviours.class_m.model.alignm_dist_thres,
                "coef_cohesion": cfg.behaviours.class_m.model.cohe_w,
                "d_cohe": cfg.behaviours.class_m.model.cohe_dist_thres,
                "coef_separation": cfg.behaviours.class_m.model.separ_w,
                "d_sepa": cfg.behaviours.class_m.model.separ_dist_thres,
                "f_lim": cfg.behaviours.class_m.max_abs_f,
                "v_lim": cfg.behaviours.class_m.max_abs_sp,
                "v_const": cfg.behaviours.class_m.init_speed,
                "eps": cfg.behaviours.class_m.eps,
                "walls_b": cfg.behaviours.class_m.walls_b,
                "store_interm_data": cfg.behaviours.class_m.storing_data,
                "interm_data_names": cfg.behaviours.class_m.model.interm_data_names,
            },
        )

        if cfg.behaviours.class_m.storing_data:
            boids_rey, _ = out_step
        else:
            boids_rey = out_step

        boids_buffer.append(boids_rey)

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


def unit_t7(output_dir: str, state: dict, cfg: DictConfig) -> None:
    """
    Using the alignment + cohesion + separation forces Reynolds flocking model
    """
    boids_buffer = [state["boids"]]
    os.makedirs(output_dir + f"/unit_t7/", mode=0o777, exist_ok=True)

    # Simulation loop
    boids_rey = boids_buffer[0]
    for _ in range(cfg.behaviours.class_m.steps):
        out_step = original_reynolds_step(
            boids_rey,
            {
                "box_init": cfg.behaviours.class_m.box_init,
                "max_dist": cfg.behaviours.class_m.model.alignm_dist_thres,
                "box_size": cfg.behaviours.class_m.box_size,
                "dt": cfg.behaviours.class_m.dt,
                "inner_loop": cfg.behaviours.class_m.inner_loop,
                "sim_speed": cfg.behaviours.class_m.sim_speed,
                "max_speed": cfg.behaviours.class_m.max_abs_sp,
                "coef_alignment": cfg.behaviours.class_m.model.alignm_w,
                "d_align": cfg.behaviours.class_m.model.alignm_dist_thres,
                "coef_cohesion": cfg.behaviours.class_m.model.cohe_w,
                "d_cohe": cfg.behaviours.class_m.model.cohe_dist_thres,
                "coef_separation": cfg.behaviours.class_m.model.separ_w,
                "d_sepa": cfg.behaviours.class_m.model.separ_dist_thres,
                "f_lim": cfg.behaviours.class_m.max_abs_f,
                "v_lim": cfg.behaviours.class_m.max_abs_sp,
                "v_const": cfg.behaviours.class_m.init_speed,
                "eps": cfg.behaviours.class_m.eps,
                "walls_b": cfg.behaviours.class_m.walls_b,
                "store_interm_data": cfg.behaviours.class_m.storing_data,
                "interm_data_names": cfg.behaviours.class_m.model.interm_data_names,
            },
        )

        if cfg.behaviours.class_m.storing_data:
            boids_rey, _ = out_step
        else:
            boids_rey = out_step

        boids_buffer.append(boids_rey)

        # Checking that all agents' speed is constant
        assert np.any(
            np.abs(
                np.linalg.norm(boids_rey.X_dot, axis=1)
                - cfg.behaviours.class_m.init_speed
            )
            < 1e-1
        ), "All agent's speeds should be constant."

    # Visualization for reynolds forces
    total_f_f, _, _, _ = original_reynolds_force(
        boids_buffer[-1],
        {
            "box_init": cfg.behaviours.class_m.box_init,
            "max_dist": cfg.behaviours.class_m.model.alignm_dist_thres,
            "box_size": cfg.behaviours.class_m.box_size,
            "dt": cfg.behaviours.class_m.dt,
            "inner_loop": cfg.behaviours.class_m.inner_loop,
            "sim_speed": cfg.behaviours.class_m.sim_speed,
            "max_speed": cfg.behaviours.class_m.max_abs_sp,
            "coef_alignment": cfg.behaviours.class_m.model.alignm_w,
            "d_align": cfg.behaviours.class_m.model.alignm_dist_thres,
            "coef_cohesion": cfg.behaviours.class_m.model.cohe_w,
            "d_cohe": cfg.behaviours.class_m.model.cohe_dist_thres,
            "coef_separation": cfg.behaviours.class_m.model.separ_w,
            "d_sepa": cfg.behaviours.class_m.model.separ_dist_thres,
            "f_lim": cfg.behaviours.class_m.max_abs_f,
            "v_lim": cfg.behaviours.class_m.max_abs_sp,
            "v_const": cfg.behaviours.class_m.init_speed,
            "eps": cfg.behaviours.class_m.eps,
            "walls_b": cfg.behaviours.class_m.walls_b,
            "store_interm_data": cfg.behaviours.class_m.storing_data,
            "interm_data_names": cfg.behaviours.class_m.model.interm_data_names,
        },
    )
    total_f_f_x, total_f_f_y = np.split(total_f_f, 2, axis=1)
    visualizing_boids_force(
        boids_buffer[-1].X,
        (
            (
                total_f_f_x,
                total_f_f_y,
                "blue",
                cfg.behaviours.class_m.box_size // 3,
                "Total force",
            ),
        ),
        output_dir + f"/unit_t7/boids_forces.png",
        "Reynolds Boids Forces final state",
    )
    # Exporting the frames to a video for visual inspection
    render_boids_gif(
        cfg.behaviours.class_m.box_size,
        boids_buffer[-(len(boids_buffer) // 20) :],
        output_dir + "/unit_t7/frames.gif",
        cfg.behaviours.agent_trace_steps,
        cfg.behaviours.fps_render,
        cfg.behaviours.class_m.model.alignm_dist_thres,
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
            "out_name": output_dir + "/unit_t7/sakana_frames.png",
            "fps": cfg.behaviours.fps_render,
        },
        time_sampling=(8, True),
    )


def unit_t8(vic_state: dict, rey_state: dict) -> None:
    """
    Testing state mapping from the Vicsek flocking model to Reynolds (and back)
    """
    vic_angles = vic_state["boids"].Theta
    rey_vel_vecs = rey_state["boids"].X_dot
    rey_vel_vecs[
        np.random.randint(0, rey_vel_vecs.shape[0], size=3), 0
    ] *= -1  # to get `from_vec_to_angle` edge cases

    new_rey_unit_vel_vecs = from_angle_to_vec(vic_angles)
    assert (
        new_rey_unit_vel_vecs.shape[0] == vic_angles.shape[0]
    ), "Should be the same number of rows"

    new_vic_angles = from_vec_to_angle(rey_vel_vecs)
    assert (
        new_vic_angles.shape[0] == rey_vel_vecs.shape[0]
    ), "Should be the same number of rows"


def unit_t9(output_dir: str, state: dict, cfg: DictConfig) -> None:
    """
    Using the 'sort-of' alignment force Sicsek flocking model
    """
    boids_buffer = [state["boids"]]
    os.makedirs(output_dir + f"/unit_t9/", mode=0o777, exist_ok=True)

    # Simulation loop
    boids_vic = boids_buffer[0]
    for _ in range(cfg.behaviours.class_m.steps):
        out_step = original_vicsek_step(
            deepcopy(boids_vic),
            {
                "box_init": cfg.behaviours.class_m.box_init,
                "max_dist": cfg.behaviours.class_m.model.alignm_dist_thres,
                "box_size": cfg.behaviours.class_m.box_size,
                "dt": cfg.behaviours.class_m.dt,
                "inner_loop": cfg.behaviours.class_m.inner_loop,
                "sim_speed": cfg.behaviours.class_m.sim_speed,
                "d_align": cfg.behaviours.class_m.model.alignm_dist_thres,
                "max_speed": cfg.behaviours.class_m.max_abs_sp,
                "coef_alignment": cfg.behaviours.class_m.model.alignm_w,
                "noise": cfg.behaviours.class_m.model.noise,
                "v_const": cfg.behaviours.class_m.model.speed_vic,
                "eps": cfg.behaviours.class_m.eps,
                "walls_b": cfg.behaviours.class_m.walls_b,
                "store_interm_data": cfg.behaviours.class_m.storing_data,
                "interm_data_names": cfg.behaviours.class_m.model.interm_data_names,
            },
        )

        if cfg.behaviours.class_m.storing_data:
            boids_vic, _ = out_step
        else:
            boids_vic = out_step

        boids_buffer.append(boids_vic)

        # Checking that all agents' speed is constant
        assert np.any(
            np.abs(
                np.linalg.norm(boids_vic.X_dot, axis=1)
                - cfg.behaviours.class_m.init_speed
            )
            < 1e-1
        ), "All agent's speeds should be constant."

    # Visualization for vicsek forces
    total_f_f, _ = original_vicsek_force(
        boids_buffer[-1],
        {
            "box_init": cfg.behaviours.class_m.box_init,
            "max_dist": cfg.behaviours.class_m.model.alignm_dist_thres,
            "box_size": cfg.behaviours.class_m.box_size,
            "dt": cfg.behaviours.class_m.dt,
            "inner_loop": cfg.behaviours.class_m.inner_loop,
            "sim_speed": cfg.behaviours.class_m.sim_speed,
            "d_align": cfg.behaviours.class_m.model.alignm_dist_thres,
            "max_speed": cfg.behaviours.class_m.max_abs_sp,
            "coef_alignment": cfg.behaviours.class_m.model.alignm_w,
            "noise": cfg.behaviours.class_m.model.noise,
            "v_const": cfg.behaviours.class_m.model.speed_vic,
            "eps": cfg.behaviours.class_m.eps,
            "walls_b": cfg.behaviours.class_m.walls_b,
            "store_interm_data": cfg.behaviours.class_m.storing_data,
            "interm_data_names": cfg.behaviours.class_m.model.interm_data_names,
        },
    )
    total_f_f_xy = cfg.behaviours.class_m.model.speed_vic * from_angle_to_vec(total_f_f)
    total_f_f_x, total_f_f_y = np.split(total_f_f_xy, 2, axis=1)
    visualizing_boids_force(
        boids_buffer[-1].X,
        (
            (
                total_f_f_x,
                total_f_f_y,
                "blue",
                cfg.behaviours.class_m.box_size // 3,
                "Total force",
            ),
        ),
        output_dir + f"/unit_t9/boids_forces.png",
        "Vicsek Boids Forces final state",
    )
    # Exporting the frames to a video for visual inspection
    render_boids_gif(
        cfg.behaviours.class_m.box_size,
        boids_buffer[-(len(boids_buffer) // 20) :],
        output_dir + "/unit_t9/frames.gif",
        cfg.behaviours.agent_trace_steps,
        cfg.behaviours.fps_render,
        cfg.behaviours.class_m.model.alignm_dist_thres,
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
            "out_name": output_dir + "/unit_t9/sakana_frames.png",
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

    # Reynolds tests
    cfg.behaviours.class_m = cfg.behaviours.flocking
    cfg.behaviours.class_m.model = cfg.behaviours.flocking.reynolds
    boids_rey = initialize_boids_flocking(
        cfg.behaviours.class_m.box_size,
        cfg.behaviours.class_m.boid_count,
        cfg.behaviours.class_m.dim,
        cfg.behaviours.class_m.seed,
        cfg.behaviours.class_m.init_speed,
        cfg.behaviours.class_m.model.name,
    )
    state1 = {"boids": deepcopy(boids_rey)}
    unit_t1(out_dir, state1, cfg)
    state2 = {"boids": deepcopy(boids_rey)}
    unit_t2(out_dir, state2, cfg)
    state3 = {"boids": deepcopy(boids_rey)}
    unit_t3(out_dir, state3, cfg)
    state4 = {"boids": deepcopy(boids_rey)}
    unit_t4(out_dir, state4, cfg)
    state5 = {"boids": deepcopy(boids_rey)}
    unit_t5(out_dir, state5, cfg)
    state6 = {"boids": deepcopy(boids_rey)}
    unit_t6(out_dir, state6, cfg)
    state7 = {"boids": deepcopy(boids_rey)}
    unit_t7(out_dir, state7, cfg)
    # Exporting the updated config
    with open(out_dir + "/.hydra/runtime_config_reynolds.yaml", "w") as f:
        OmegaConf.save(config=cfg, f=f.name)

    # Vicsek tests
    cfg.behaviours.class_m.model = cfg.behaviours.flocking.vicsek
    boids_vic = initialize_boids_flocking(
        cfg.behaviours.class_m.box_size,
        cfg.behaviours.class_m.boid_count,
        cfg.behaviours.class_m.dim,
        cfg.behaviours.class_m.seed,
        cfg.behaviours.class_m.init_speed,
        cfg.behaviours.class_m.model.name,
    )
    state8 = {"boids": deepcopy(boids_vic)}
    unit_t8(state8, {"boids": deepcopy(boids_rey)})
    state9 = {"boids": deepcopy(boids_vic)}
    unit_t9(out_dir, state9, cfg)
    # Exporting the updated config
    with open(out_dir + "/.hydra/runtime_config_vicsek.yaml", "w") as f:
        OmegaConf.save(config=cfg, f=f.name)


if __name__ == "__main__":
    main()
