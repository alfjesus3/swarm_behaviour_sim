import hydra
from omegaconf import DictConfig, OmegaConf
import logging

import os
import sys
from copy import deepcopy

from swarm_hydra.entry_point import *
from swarm_hydra.behaviours.utils_behaviours import *
from swarm_hydra.behaviours.flocking import *
from swarm_hydra.behaviours.aggregation import *
from swarm_hydra.behaviours.dispersion import *
from swarm_hydra.behaviours.random_walk import *

# A logger for this file
log = logging.getLogger(__name__)


def gen_reynolds_data(output_dir: str, state: dict, cfg: DictConfig):
    """
    Using the alignment + cohesion + separation forces Reynolds flocking model
    """
    os.makedirs(output_dir + "/" + cfg.experiment_name, mode=0o777, exist_ok=True)

    # Simulation loop
    boids_buffer = [state["boids"]]
    boids_rey = boids_buffer[0]
    for it in range(cfg.behaviours.class_m.steps):
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
            boids_rey, interm_data = out_step
            interm_data[cfg.behaviours.class_m.model.interm_data_names[0]] = it + 1
            interm_data[cfg.behaviours.class_m.model.interm_data_names[1]] = (
                boids_buffer[-1].X
            )
            interm_data[cfg.behaviours.class_m.model.interm_data_names[2]] = (
                boids_buffer[-1].X_dot
            )
            exported_to_hdf5 = export_to_hdf5(
                interm_data,
                output_dir + "/" + cfg.experiment_name + "/dataset.hdf5",
                cfg.behaviours.class_m.steps + 1,
            )
            if not exported_to_hdf5:
                log.info(
                    f"[WARNING]It was not successfully exported to HDF5 on iter{it}."
                )
        else:
            boids_rey = out_step

        boids_buffer.append(boids_rey)

    log.info(f'Fully exported to {output_dir+"/"+cfg.experiment_name+"/dataset.hdf5"}.')

    # Exporting the frames to a video for visual inspection
    render_boids_gif(
        cfg.behaviours.class_m.box_size,
        deepcopy(boids_buffer[-(len(boids_buffer) // 20) :]),
        output_dir + "/" + cfg.experiment_name + "/frames.gif",
        cfg.behaviours.agent_trace_steps,
        cfg.behaviours.fps_render,
        cfg.behaviours.class_m.model.alignm_dist_thres,
        cfg.behaviours.class_m.init_speed,
    )
    render_network_gif(
        cfg.behaviours.class_m.box_size,
        deepcopy(boids_buffer[-(len(boids_buffer) // 20) :]),
        output_dir + "/" + cfg.experiment_name + "/graph_frames.gif",
        cfg.behaviours.fps_render,
        cfg.behaviours.class_m.model.alignm_dist_thres,
        cfg.behaviours.class_m.walls_b,
    )
    render_boids_sakana(
        deepcopy(boids_buffer),
        {
            "space_size": 1.0,
            "img_size": int(cfg.behaviours.class_m.box_size),
            "bird_render_size": 0.007,  # 0.0015,  # 0.015,
            "bird_render_sharpness": 40.0,
            "red_boids": [],
            "v_const": cfg.behaviours.class_m.init_speed,
            "out_name": output_dir + "/" + cfg.experiment_name + "/sakana_frames.png",
            "fps": cfg.behaviours.fps_render,
        },
        time_sampling=(8, True),
    )


def gen_vicsek_data(output_dir: str, state: dict, cfg: DictConfig):
    """
    Using the Vicsek flocking model
    """
    os.makedirs(output_dir + "/" + cfg.experiment_name, mode=0o777, exist_ok=True)

    # Simulation loop
    boids_buffer = [state["boids"]]
    boids_vic = boids_buffer[0]
    for it in range(cfg.behaviours.class_m.steps):
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
            boids_vic, interm_data = out_step
            interm_data[cfg.behaviours.class_m.model.interm_data_names[0]] = it + 1
            interm_data[cfg.behaviours.class_m.model.interm_data_names[1]] = (
                boids_buffer[-1].X
            )
            interm_data[cfg.behaviours.class_m.model.interm_data_names[2]] = (
                boids_buffer[-1].X_dot
            )
            interm_data[cfg.behaviours.class_m.model.interm_data_names[3]] = (
                boids_buffer[-1].Theta
            )
            exported_to_hdf5 = export_to_hdf5(
                interm_data,
                output_dir + "/" + cfg.experiment_name + "/dataset.hdf5",
                cfg.behaviours.class_m.steps + 1,
            )
            if not exported_to_hdf5:
                log.info(
                    f"[WARNING]It was not successfully exported to HDF5 on iter{it}."
                )
        else:
            boids_vic = out_step

        boids_buffer.append(boids_vic)

    log.info(f'Fully exported to {output_dir+"/"+cfg.experiment_name+"/dataset.hdf5"}.')

    # Exporting the frames to a video for visual inspection
    render_boids_gif(
        cfg.behaviours.class_m.box_size,
        deepcopy(boids_buffer[-(len(boids_buffer) // 20) :]),
        output_dir + "/" + cfg.experiment_name + "/frames.gif",
        cfg.behaviours.agent_trace_steps,
        cfg.behaviours.fps_render,
        cfg.behaviours.class_m.model.alignm_dist_thres,
        cfg.behaviours.class_m.init_speed,
    )
    render_network_gif(
        cfg.behaviours.class_m.box_size,
        deepcopy(boids_buffer[-(len(boids_buffer) // 20) :]),
        output_dir + "/" + cfg.experiment_name + "/graph_frames.gif",
        cfg.behaviours.fps_render,
        cfg.behaviours.class_m.model.alignm_dist_thres,
        cfg.behaviours.class_m.walls_b,
    )
    render_boids_sakana(
        deepcopy(boids_buffer),
        {
            "space_size": 1.0,
            "img_size": int(cfg.behaviours.class_m.box_size),
            "bird_render_size": 0.007,  # 0.0015,  # 0.015,
            "bird_render_sharpness": 40.0,
            "red_boids": [],
            "v_const": cfg.behaviours.class_m.init_speed,
            "out_name": output_dir + "/" + cfg.experiment_name + "/sakana_frames.png",
            "fps": cfg.behaviours.fps_render,
        },
        time_sampling=(8, True),
    )


def gen_flocking_data(cfg, out_dir, baseline_experiment_name):
    """For flocking behavior dataset generation."""

    for curr_seed in cfg.seeds_per_exp_rep:
        cfg.behaviours.class_m.seed = curr_seed
        reset_seeds(cfg.behaviours.class_m.seed)
        cfg.experiment_name = (
            baseline_experiment_name
            + f"_{cfg.behaviour}_{cfg.behaviours.class_m.model.name}_exper_rep{cfg.behaviours.class_m.seed}"
        )

        if cfg.behaviours.class_m.model.name == cfg.behaviours.class_m.reynolds.name:
            boids_rey = initialize_boids_flocking(
                cfg.behaviours.class_m.box_size,
                cfg.behaviours.class_m.boid_count,
                cfg.behaviours.class_m.dim,
                cfg.behaviours.class_m.seed,
                cfg.behaviours.class_m.init_speed,
                cfg.behaviours.class_m.model.name,
            )
            init_rey_state = {"boids": deepcopy(boids_rey)}
            gen_reynolds_data(out_dir, init_rey_state, cfg)
        elif cfg.behaviours.class_m.model.name == cfg.behaviours.class_m.vicsek.name:
            cfg.behaviours.class_m.model.alignm_dist_thres = (
                cfg.behaviours.class_m.box_size / 10
            )  # 30  #

            boids_vic = initialize_boids_flocking(
                cfg.behaviours.class_m.box_size,
                cfg.behaviours.class_m.boid_count,
                cfg.behaviours.class_m.dim,
                cfg.behaviours.class_m.seed,
                cfg.behaviours.class_m.init_speed,
                cfg.behaviours.class_m.model.name,
            )
            init_vic_state = {"boids": deepcopy(boids_vic)}
            gen_vicsek_data(out_dir, init_vic_state, cfg)
        else:
            raise NotImplementedError

        # Exporting the updated config
        with open(
            out_dir + f"/.hydra/{cfg.experiment_name}_runtime_config.yaml", "w"
        ) as f:
            OmegaConf.save(config=cfg, f=f.name)


def gen_aggregation_data(cfg: OmegaConf, out_dir: str, baseline_experiment_name: str):
    """For aggregation behavior dataset generation."""

    for curr_seed in cfg.seeds_per_exp_rep:
        cfg.behaviours.class_m.seed = curr_seed
        reset_seeds(cfg.behaviours.class_m.seed)
        cfg.experiment_name = (
            baseline_experiment_name
            + f"_{cfg.behaviour}_{cfg.behaviours.class_m.model.name}_exper_rep{cfg.behaviours.class_m.seed}"
        )
        os.makedirs(out_dir + "/" + cfg.experiment_name, mode=0o777, exist_ok=True)

        boids_aggreg = initialize_boids_aggregation(
            cfg.behaviours.class_m.box_size,
            cfg.behaviours.class_m.boid_count,
            cfg.behaviours.class_m.dim,
            cfg.behaviours.class_m.seed,
            cfg.behaviours.class_m.init_speed,
        )

        # Simulation loop
        boids_buffer = [boids_aggreg]
        for it in range(cfg.behaviours.class_m.steps):
            out_step = probabilistic_aggregation_step(
                boids_aggreg,
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
                    "attra_w": cfg.behaviours.class_m.model.attra_w,
                    "min_dist": cfg.behaviours.class_m.model.min_dist,
                    "max_dist": cfg.behaviours.class_m.model.max_dist,
                    "max_long_vel": cfg.behaviours.class_m.max_long_vel,
                    "max_rot_vel": cfg.behaviours.class_m.max_rot_vel,
                },
            )

            if cfg.behaviours.class_m.storing_data:
                boids_aggreg, interm_data = out_step
                interm_data[cfg.behaviours.class_m.model.interm_data_names[0]] = it + 1
                interm_data[cfg.behaviours.class_m.model.interm_data_names[1]] = (
                    boids_buffer[-1].X
                )
                interm_data[cfg.behaviours.class_m.model.interm_data_names[2]] = (
                    boids_buffer[-1].X_dot
                )
                exported_to_hdf5 = export_to_hdf5(
                    interm_data,
                    out_dir + "/" + cfg.experiment_name + "/dataset.hdf5",
                    cfg.behaviours.class_m.steps + 1,
                )
                if not exported_to_hdf5:
                    log.info(
                        f"[WARNING]It was not successfully exported to HDF5 on iter{it}."
                    )
            else:
                boids_aggreg = out_step

            boids_buffer.append(boids_aggreg)

        log.info(
            f'Fully exported to {out_dir+"/"+cfg.experiment_name+"/dataset.hdf5"}.'
        )

        # Exporting the updated config
        with open(
            out_dir + f"/.hydra/{cfg.experiment_name}_runtime_config.yaml", "w"
        ) as f:
            OmegaConf.save(config=cfg, f=f.name)

        # Exporting the frames to a video for visual inspection
        render_boids_gif(
            cfg.behaviours.class_m.box_size,
            deepcopy(boids_buffer[-(len(boids_buffer) // 20) :]),
            out_dir + "/" + cfg.experiment_name + "/frames.gif",
            cfg.behaviours.agent_trace_steps,
            cfg.behaviours.fps_render,
            cfg.behaviours.class_m.model.max_dist,
            cfg.behaviours.class_m.init_speed,
        )
        render_network_gif(
            cfg.behaviours.class_m.box_size,
            deepcopy(boids_buffer[-(len(boids_buffer) // 20) :]),
            out_dir + "/" + cfg.experiment_name + "/graph_frames.gif",
            cfg.behaviours.fps_render,
            cfg.behaviours.class_m.model.max_dist,
            cfg.behaviours.class_m.walls_b,
        )
        render_boids_sakana(
            deepcopy(boids_buffer),
            {
                "space_size": 1.0,
                "img_size": int(cfg.behaviours.class_m.box_size),
                "bird_render_size": 0.007,  # 0.0015,  # 0.015,
                "bird_render_sharpness": 40.0,
                "red_boids": [],
                "v_const": cfg.behaviours.class_m.init_speed,
                "out_name": out_dir + "/" + cfg.experiment_name + "/sakana_frames.png",
                "fps": cfg.behaviours.fps_render,
            },
            time_sampling=(8, True),
        )


def gen_dispersion_data(cfg: OmegaConf, out_dir: str, baseline_experiment_name: str):
    """For dispersion behavior dataset generation."""

    for curr_seed in cfg.seeds_per_exp_rep:
        cfg.behaviours.class_m.seed = curr_seed
        reset_seeds(cfg.behaviours.class_m.seed)
        cfg.experiment_name = (
            baseline_experiment_name
            + f"_{cfg.behaviour}_{cfg.behaviours.class_m.model.name}_exper_rep{cfg.behaviours.class_m.seed}"
        )
        os.makedirs(out_dir + "/" + cfg.experiment_name, mode=0o777, exist_ok=True)

        boids_disper = initialize_boids_disperation(
            cfg.behaviours.class_m.box_size,
            cfg.behaviours.class_m.boid_count,
            cfg.behaviours.class_m.dim,
            cfg.behaviours.class_m.seed,
            cfg.behaviours.class_m.init_speed,
        )

        # Simulation loop
        boids_buffer = [boids_disper]
        for it in range(cfg.behaviours.class_m.steps):
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
                boids_disper, interm_data = out_step
                interm_data[cfg.behaviours.class_m.model.interm_data_names[0]] = it + 1
                interm_data[cfg.behaviours.class_m.model.interm_data_names[1]] = (
                    boids_buffer[-1].X
                )
                interm_data[cfg.behaviours.class_m.model.interm_data_names[2]] = (
                    boids_buffer[-1].X_dot
                )
                exported_to_hdf5 = export_to_hdf5(
                    interm_data,
                    out_dir + "/" + cfg.experiment_name + "/dataset.hdf5",
                    cfg.behaviours.class_m.steps + 1,
                )
                if not exported_to_hdf5:
                    log.info(
                        f"[WARNING]It was not successfully exported to HDF5 on iter{it}."
                    )
            else:
                boids_disper = out_step

            boids_buffer.append(boids_disper)

        log.info(
            f'Fully exported to {out_dir+"/"+cfg.experiment_name+"/dataset.hdf5"}.'
        )

        # Exporting the updated config
        with open(
            out_dir + f"/.hydra/{cfg.experiment_name}_runtime_config.yaml", "w"
        ) as f:
            OmegaConf.save(config=cfg, f=f.name)

        # Exporting the frames to a video for visual inspection
        render_boids_gif(
            cfg.behaviours.class_m.box_size,
            deepcopy(boids_buffer[-(len(boids_buffer) // 20) :]),
            out_dir + "/" + cfg.experiment_name + "/frames.gif",
            cfg.behaviours.agent_trace_steps,
            cfg.behaviours.fps_render,
            cfg.behaviours.class_m.model.max_dist,
            cfg.behaviours.class_m.init_speed,
        )
        render_network_gif(
            cfg.behaviours.class_m.box_size,
            deepcopy(boids_buffer[-(len(boids_buffer) // 20) :]),
            out_dir + "/" + cfg.experiment_name + "/graph_frames.gif",
            cfg.behaviours.fps_render,
            cfg.behaviours.class_m.model.max_dist,
            cfg.behaviours.class_m.walls_b,
        )
        render_boids_sakana(
            deepcopy(boids_buffer),
            {
                "space_size": 1.0,
                "img_size": int(cfg.behaviours.class_m.box_size),
                "bird_render_size": 0.007,  # 0.0015,  # 0.015,
                "bird_render_sharpness": 40.0,
                "red_boids": [],
                "v_const": cfg.behaviours.class_m.init_speed,
                "out_name": out_dir + "/" + cfg.experiment_name + "/sakana_frames.png",
                "fps": cfg.behaviours.fps_render,
            },
            time_sampling=(8, True),
        )


def gen_ballistic_data(output_dir: str, state: dict, cfg: DictConfig):
    """
    Using the Ballistic Motion Model
    """
    os.makedirs(output_dir + "/" + cfg.experiment_name, mode=0o777, exist_ok=True)

    # Simulation loop
    boids_buffer = [state["boids"]]
    boids_ballistic = boids_buffer[0]
    for it in range(cfg.behaviours.class_m.steps):
        out_step = ballistic_random_walk_step(
            boids_ballistic,
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
            boids_ballistic, interm_data = out_step
            interm_data[cfg.behaviours.class_m.model.interm_data_names[0]] = it + 1
            interm_data[cfg.behaviours.class_m.model.interm_data_names[1]] = (
                boids_buffer[-1].X
            )
            interm_data[cfg.behaviours.class_m.model.interm_data_names[2]] = (
                boids_buffer[-1].X_dot
            )
            exported_to_hdf5 = export_to_hdf5(
                interm_data,
                output_dir + "/" + cfg.experiment_name + "/dataset.hdf5",
                cfg.behaviours.class_m.steps + 1,
            )
            if not exported_to_hdf5:
                log.info(
                    f"[WARNING]It was not successfully exported to HDF5 on iter{it}."
                )
        else:
            boids_ballistic = out_step

        boids_buffer.append(boids_ballistic)

    log.info(f'Fully exported to {output_dir+"/"+cfg.experiment_name+"/dataset.hdf5"}.')

    # Exporting the updated config
    with open(
        output_dir + f"/.hydra/{cfg.experiment_name}_runtime_config.yaml", "w"
    ) as f:
        OmegaConf.save(config=cfg, f=f.name)

    # Exporting the frames to a video for visual inspection
    render_boids_gif(
        cfg.behaviours.class_m.box_size,
        deepcopy(boids_buffer[-(len(boids_buffer) // 20) :]),
        output_dir + "/" + cfg.experiment_name + "/frames.gif",
        cfg.behaviours.agent_trace_steps,
        cfg.behaviours.fps_render,
        cfg.behaviours.class_m.model.max_dist,
        cfg.behaviours.class_m.init_speed,
    )
    render_network_gif(
        cfg.behaviours.class_m.box_size,
        deepcopy(boids_buffer[-(len(boids_buffer) // 20) :]),
        output_dir + "/" + cfg.experiment_name + "/graph_frames.gif",
        cfg.behaviours.fps_render,
        cfg.behaviours.class_m.model.max_dist,
        cfg.behaviours.class_m.walls_b,
    )
    render_boids_sakana(
        deepcopy(boids_buffer),
        {
            "space_size": 1.0,
            "img_size": int(cfg.behaviours.class_m.box_size),
            "bird_render_size": 0.007,  # 0.0015,  # 0.015,
            "bird_render_sharpness": 40.0,
            "red_boids": [],
            "v_const": cfg.behaviours.class_m.init_speed,
            "out_name": output_dir + "/" + cfg.experiment_name + "/sakana_frames.png",
            "fps": cfg.behaviours.fps_render,
        },
        time_sampling=(8, True),
    )


def gen_brownian_data(output_dir: str, state: dict, cfg: DictConfig):
    """
    Using the Brownian Motion Model
    """
    os.makedirs(output_dir + "/" + cfg.experiment_name, mode=0o777, exist_ok=True)

    # Simulation loop
    boids_buffer = [state["boids"]]
    boids_brownian = boids_buffer[0]
    for it in range(cfg.behaviours.class_m.steps):
        out_step = brownian_random_walk_step(
            boids_brownian,
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
            boids_brownian, interm_data = out_step
            interm_data[cfg.behaviours.class_m.model.interm_data_names[0]] = it + 1
            interm_data[cfg.behaviours.class_m.model.interm_data_names[1]] = (
                boids_buffer[-1].X
            )
            interm_data[cfg.behaviours.class_m.model.interm_data_names[2]] = (
                boids_buffer[-1].X_dot
            )
            interm_data[cfg.behaviours.class_m.model.interm_data_names[3]] = (
                boids_buffer[-1].Counter
            )
            exported_to_hdf5 = export_to_hdf5(
                interm_data,
                output_dir + "/" + cfg.experiment_name + "/dataset.hdf5",
                cfg.behaviours.class_m.steps + 1,
            )
            if not exported_to_hdf5:
                log.info(
                    f"[WARNING]It was not successfully exported to HDF5 on iter{it}."
                )
        else:
            boids_brownian = out_step

        boids_buffer.append(boids_brownian)

    log.info(f'Fully exported to {output_dir+"/"+cfg.experiment_name+"/dataset.hdf5"}.')

    # Exporting the updated config
    with open(
        output_dir + f"/.hydra/{cfg.experiment_name}_runtime_config.yaml", "w"
    ) as f:
        OmegaConf.save(config=cfg, f=f.name)

    # Exporting the frames to a video for visual inspection
    render_boids_gif(
        cfg.behaviours.class_m.box_size,
        deepcopy(boids_buffer[-(len(boids_buffer) // 20) :]),
        output_dir + "/" + cfg.experiment_name + "/frames.gif",
        cfg.behaviours.agent_trace_steps,
        cfg.behaviours.fps_render,
        cfg.behaviours.class_m.model.max_dist,
        cfg.behaviours.class_m.init_speed,
    )
    render_network_gif(
        cfg.behaviours.class_m.box_size,
        deepcopy(boids_buffer[-(len(boids_buffer) // 20) :]),
        output_dir + "/" + cfg.experiment_name + "/graph_frames.gif",
        cfg.behaviours.fps_render,
        cfg.behaviours.class_m.model.max_dist,
        cfg.behaviours.class_m.walls_b,
    )
    render_boids_sakana(
        deepcopy(boids_buffer),
        {
            "space_size": 1.0,
            "img_size": int(cfg.behaviours.class_m.box_size),
            "bird_render_size": 0.007,  # 0.0015,  # 0.015,
            "bird_render_sharpness": 40.0,
            "red_boids": [],
            "v_const": cfg.behaviours.class_m.init_speed,
            "out_name": output_dir + "/" + cfg.experiment_name + "/sakana_frames.png",
            "fps": cfg.behaviours.fps_render,
        },
        time_sampling=(8, True),
    )


def gen_random_walk_data(cfg: OmegaConf, out_dir: str, baseline_experiment_name: str):
    """For random walk behavior dataset generation."""

    for curr_seed in cfg.seeds_per_exp_rep:
        cfg.behaviours.class_m.seed = curr_seed
        reset_seeds(cfg.behaviours.class_m.seed)
        cfg.experiment_name = (
            baseline_experiment_name
            + f"_{cfg.behaviour}_{cfg.behaviours.class_m.model.name}_exper_rep{cfg.behaviours.class_m.seed}"
        )

        if cfg.behaviours.class_m.model.name == cfg.behaviours.class_m.ballistic.name:
            boids_ballistic = initialize_boids_random_walk(
                cfg.behaviours.class_m.box_size,
                cfg.behaviours.class_m.boid_count,
                cfg.behaviours.class_m.dim,
                cfg.behaviours.class_m.seed,
                cfg.behaviours.class_m.init_speed,
                cfg.behaviours.class_m.model.name,
                (None, None),
            )
            init_ballistic_state = {"boids": deepcopy(boids_ballistic)}
            gen_ballistic_data(out_dir, init_ballistic_state, cfg)
        elif cfg.behaviours.class_m.model.name == cfg.behaviours.class_m.brownian.name:
            boids_brownian = initialize_boids_random_walk(
                cfg.behaviours.class_m.box_size,
                cfg.behaviours.class_m.boid_count,
                cfg.behaviours.class_m.dim,
                cfg.behaviours.class_m.seed,
                cfg.behaviours.class_m.init_speed,
                cfg.behaviours.class_m.model.name,
                (
                    cfg.behaviours.class_m.model.mu,
                    cfg.behaviours.class_m.model.beta_scale,
                ),
            )
            init_brownian_state = {"boids": deepcopy(boids_brownian)}
            gen_brownian_data(out_dir, init_brownian_state, cfg)
        else:
            raise NotImplementedError

        # Exporting the updated config
        with open(
            out_dir + f"/.hydra/{cfg.experiment_name}_runtime_config.yaml", "w"
        ) as f:
            OmegaConf.save(config=cfg, f=f.name)


@hydra.main(config_path="../configs", config_name="config", version_base="1.1")
def main(cfg: DictConfig):
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    hydra_cfg["runtime"]["output_dir"]
    out_dir = hydra_cfg["runtime"]["output_dir"]

    # Redirect stdout & stderr
    sys.stdout = LoggerWriter(log, logging.INFO)
    sys.stderr = LoggerWriter(log, logging.ERROR)

    if cfg.behaviours.class_m.name == cfg.behaviours.flocking.name:
        cfg.behaviours.class_m.model = cfg.behaviours.class_m.reynolds
        original_experiment_name = deepcopy(cfg.experiment_name)
        gen_flocking_data(cfg, out_dir, cfg.experiment_name)
        cfg.behaviours.class_m.model = cfg.behaviours.class_m.vicsek
        cfg.experiment_name = original_experiment_name
        gen_flocking_data(cfg, out_dir, cfg.experiment_name)
    elif cfg.behaviours.class_m.name == cfg.behaviours.aggregation.name:
        cfg.behaviours.class_m.model = cfg.behaviours.aggregation.probab
        gen_aggregation_data(cfg, out_dir, cfg.experiment_name)
    elif cfg.behaviours.class_m.name == cfg.behaviours.dispersion.name:
        cfg.behaviours.class_m.model = cfg.behaviours.dispersion.repu_field
        gen_dispersion_data(cfg, out_dir, cfg.experiment_name)
    elif cfg.behaviours.class_m.name == cfg.behaviours.random_walk.name:
        cfg.behaviours.class_m.model = cfg.behaviours.random_walk.ballistic
        original_experiment_name = deepcopy(cfg.experiment_name)
        gen_random_walk_data(cfg, out_dir, cfg.experiment_name)
        cfg.behaviours.class_m.model = cfg.behaviours.class_m.brownian
        cfg.experiment_name = original_experiment_name
        gen_random_walk_data(cfg, out_dir, cfg.experiment_name)
    else:
        raise NotImplementedError


if __name__ == "__main__":
    main()
