import hydra
from omegaconf import DictConfig, OmegaConf
import logging

import numpy as np
import sys
import os

from swarm_hydra.entry_point import *
from swarm_hydra.metrics.utils_metrics import *
from swarm_hydra.metrics.spatial_metrics import *
from swarm_hydra.metrics.temporal_metrics import *
from swarm_hydra.metrics.proba_metrics import *
from swarm_hydra.metrics.kinematic_tree import *

# A logger for this file
log = logging.getLogger(__name__)


def test_utils_metrics(
    in_experi_dir: dir,
    curr_cfg: OmegaConf,
) -> None:
    """
    Note: these tests must read an input hdf5 to
    guarantee they are up to date with the lastest structure of the
    of the HDF5 file.
    """
    ######
    # Test the experiment folder preprocessing structure on a separate unit test too
    ######
    cfg_experi_rey, hdf5_experi_rey, reynolds_subfolders = processing_folder_structure(
        in_experi_dir, "reynolds"
    )
    reynold_states1 = hdf5_experi_rey[reynolds_subfolders[0]]
    positions1 = hdf5_experi_rey[reynolds_subfolders[0]].get("boids_pos")
    assert len(cfg_experi_rey.keys()) == (
        len(reynolds_subfolders)
    ), "Should have the same len()."
    assert len(hdf5_experi_rey.keys()) == len(
        cfg_experi_rey.keys()
    ), "Should have the same len()."
    cfg_experi_vic, hdf5_experi_vic, vicsek_subfolders = processing_folder_structure(
        in_experi_dir, "vicsek"
    )
    assert len(cfg_experi_vic.keys()) == (
        len(vicsek_subfolders)
    ), "Should have the same len()."
    assert len(hdf5_experi_vic.keys()) == len(
        cfg_experi_vic.keys()
    ), "Should have the same len()."

    ######
    # Test the CSV created, append, closing logic
    ######
    # Define and export pd.DataFrame
    test_csv = "test.csv"
    test_df = open_dataframe(test_csv, ["Col1", "Col2"])
    assert not os.path.exists(test_csv), f"There should exist no {test_csv} yet."
    for col1_dummy, col2_dummy in zip(
        np.random.randint(0, 100, size=(3,)), ["a", "b", "c"]
    ):
        test_df = append_row_to_dataframe(
            test_df, {"Col1": col1_dummy, "Col2": col2_dummy}
        )
    export_dataframe(test_df, test_csv, latex_columns=[["Col1", "Col2"]])
    assert os.path.exists(test_csv), f"There should exist a {test_csv} file."
    test_latex = test_csv.replace(".csv", f"_{0}.tex")
    assert os.path.exists(test_latex), f"There should exist a {test_latex} file."

    ######
    # Testing the sorting of arrays by defined criteria
    ######
    sorted_idxs1, _ = sort_agents_by_proximity_to_origin(
        positions1[0],
        curr_cfg.behaviours.class_m.walls_b,
        curr_cfg.behaviours.class_m.box_size,
    )
    assert (
        sorted_idxs1.shape[0] == positions1[0].shape[0]
    ), "Should have the same shape."
    update_positions10 = sort_array_by_indices(positions1[0], sorted_idxs1)
    assert (
        update_positions10.shape == positions1[0].shape
    ), "Should have the same shape."
    sorted_lst_of_idxs1, _ = sort_agents_by_proximity_to_origin(
        positions1,
        curr_cfg.behaviours.class_m.walls_b,
        curr_cfg.behaviours.class_m.box_size,
    )
    assert (
        sorted_lst_of_idxs1.shape == positions1.shape[:2]
    ), "Should have the same shape."
    update_positions1 = sort_array_by_indices(positions1, sorted_lst_of_idxs1)
    assert update_positions1.shape == positions1.shape, "Should have the same shape."
    boids_vel_arr = reynold_states1.get("boids_vels")
    to_sort_arrs = [positions1, boids_vel_arr]
    sorted_reynold_states1 = sort_arrays_by_indices(to_sort_arrs, sorted_lst_of_idxs1)
    assert len(sorted_reynold_states1) == len(
        to_sort_arrs
    ), "Should have the same shape."

    ######
    # Testing normalization of quantities
    ######
    normalized_positions1 = normalizing_quantities(
        sorted_reynold_states1[0],
        q_type="abs_position",
        params={
            "min_val": curr_cfg.behaviours.class_m.box_init,
            "max_val": curr_cfg.behaviours.class_m.box_size,
        },
    )
    assert (
        normalized_positions1.shape == positions1.shape
    ), "Should have the same shape."
    normalized_boids_vel_arr = normalizing_quantities(
        sorted_reynold_states1[1], q_type="directional", params={}
    )
    assert (
        normalized_boids_vel_arr.shape == boids_vel_arr.shape
    ), "Should have the same shape."
    normalized_reynold_states1 = normalizing_multiple_quantities(
        [positions1, boids_vel_arr],
        ["abs_position", "directional"],
        {
            "min_val": curr_cfg.behaviours.class_m.box_init,
            "max_val": curr_cfg.behaviours.class_m.box_size,
        },
    )
    assert (
        normalized_reynold_states1[0].all() == normalized_positions1.all()
    ), "Should be identical."
    assert (
        normalized_reynold_states1[1].all() == normalized_boids_vel_arr.all()
    ), "Should be identical."

    ######
    # Test combining both preprocessing steps
    ######
    proc_boids_pos, [proc_boids_vels] = preprocessing_data_for_metrics(
        positions1,
        [hdf5_experi_rey[reynolds_subfolders[0]].get("boids_vels")],
        {
            "walls_b": curr_cfg.behaviours.class_m.walls_b,
            "min_val": curr_cfg.behaviours.class_m.box_init,
            "max_val": curr_cfg.behaviours.class_m.box_size,
        },
    )
    assert proc_boids_pos.all() == normalized_positions1.all(), "Should be identical."
    assert (
        proc_boids_vels.all() == normalized_boids_vel_arr.all()
    ), "Should be identical."


def test_mse_metric(positions1: np.array, curr_cfg: OmegaConf):
    """"""
    mse_test_res1 = compute_mse_swarm_configuration(
        positions1[0],
        curr_cfg.behaviours.class_m.reynolds.alignm_dist_thres,
        curr_cfg.behaviours.class_m.walls_b,
        curr_cfg.behaviours.class_m.box_size,
    )
    assert (
        isinstance(mse_test_res1, float) and mse_test_res1 >= 0.0
    ), "Should be a positive float."
    mse_test_multi_res = []
    for swarm_config_state in positions1:
        mse_test_multi_res.append(
            compute_mse_swarm_configuration(
                swarm_config_state,
                curr_cfg.behaviours.class_m.reynolds.alignm_dist_thres,
                curr_cfg.behaviours.class_m.walls_b,
                curr_cfg.behaviours.class_m.box_size,
            )
        )
    assert len(mse_test_multi_res) == len(positions1), "Should have the same shape."


def test_area_metric(positions1: np.array, cfg1: OmegaConf):
    """"""
    area_test_res1 = compute_convex_hull_area_swarm_configuration(positions1[0])
    assert (
        isinstance(area_test_res1, float)
        and 0.0
        <= area_test_res1
        <= (cfg1.behaviours.class_m.box_init + cfg1.behaviours.class_m.box_size) ** 2
    ), "Should be a positive float <= `maximum area`."
    area_test_multi_res = []
    for swarm_config_state in positions1:
        area_test_multi_res.append(
            compute_convex_hull_area_swarm_configuration(swarm_config_state)
        )
    assert len(area_test_multi_res) == len(positions1), "Should have the same shape."


def test_polarisation_metric(velocities1: np.array):
    """"""
    polarisation_test_res1 = compute_polarisation_swarm_configuration(velocities1[0])
    assert (
        isinstance(polarisation_test_res1, float) and polarisation_test_res1 >= 0.0
    ), "Should be a positive float."
    polarisation_test_multi_res = []
    for swarm_config_state in velocities1:
        polarisation_test_multi_res.append(
            compute_polarisation_swarm_configuration(swarm_config_state)
        )
    assert len(polarisation_test_multi_res) == len(
        velocities1
    ), "Should have the same shape."


def test_combined_state_count_metric(
    positions1: np.ndarray,
    positions2: np.ndarray,
    velocities1: np.ndarray,
    velocities2: np.ndarray,
    curr_cfg: OmegaConf,
) -> None:
    """"""
    states_count1, states_count2, similarity = compute_combined_state_count_measure(
        np.concatenate((positions1, velocities1), axis=-1)[np.newaxis, :, :, :],
        np.concatenate((positions2, velocities2), axis=-1)[np.newaxis, :, :, :],
        curr_cfg.metrics.combined_state_count.num_bins,
        curr_cfg.metrics.combined_state_count.cut_off,
    )
    assert states_count1 <= (
        positions1.shape[0] * positions1.shape[1]
    ), "Should have <= len()."
    assert states_count2 <= (
        positions2.shape[0] * positions2.shape[1]
    ), "Should have <= len()."
    assert 0.0 <= similarity <= 1.0, "The similarity metric must be [0, 1]."
    log.info(f"Unique State Count (System 1): {states_count1}")
    log.info(f"Unique State Count (System 2): {states_count2}")
    log.info(f"Combined State Count Metric: {similarity:.4f}")


def test_sampled_average_state_metric(
    positions1: np.ndarray,
    positions2: np.ndarray,
    velocities1: np.ndarray,
    velocities2: np.ndarray,
    curr_cfg: OmegaConf,
) -> None:
    """"""
    sas_similarity = compute_sampled_average_state_measure(
        np.concatenate((positions1, velocities1), axis=-1)[np.newaxis, :, :, :],
        np.concatenate((positions2, velocities2), axis=-1)[np.newaxis, :, :, :],
        num_time_windows=curr_cfg.metrics.sampled_average_state.num_win,
        num_steps_sim=positions1.shape[0],
        eps=curr_cfg.metrics.sampled_average_state.eps,
    )
    assert sas_similarity >= 0.0, "The similarity metric must be strictly positive."
    log.info(f"Sampled Average State Metric: {sas_similarity:.4f}")


def test_average_local_density_metric(positions1: np.ndarray, cfg1: OmegaConf) -> None:
    avg_local_d_test_res1 = compute_average_local_density_measure(
        positions1[0],
        cfg1.behaviours.flocking.reynolds.alignm_dist_thres,
        cfg1.behaviours.flocking.box_size,
        cfg1.behaviours.flocking.walls_b,
    )
    assert isinstance(
        avg_local_d_test_res1, float
    ) and 0.0 <= avg_local_d_test_res1 <= np.sqrt(
        2 * (cfg1.behaviours.class_m.box_init + cfg1.behaviours.class_m.box_size) ** 2
    ), "Should be a positive float <= `diagonal box dims environment`."
    avg_local_d_test_multi_res = []
    for swarm_config_positions in positions1:
        avg_local_d_test_multi_res.append(
            compute_average_local_density_measure(
                swarm_config_positions,
                cfg1.behaviours.flocking.reynolds.alignm_dist_thres,
                cfg1.behaviours.flocking.box_size,
                cfg1.behaviours.flocking.walls_b,
            )
        )
    assert len(avg_local_d_test_multi_res) == len(
        positions1
    ), "Should have the same shape."


def test_beta_index_metric(positions1: np.ndarray, cfg1: OmegaConf) -> None:
    beta_idx_test_res1 = compute_beta_index_measure(
        positions1[0],
        cfg1.behaviours.flocking.box_size,
        cfg1.behaviours.flocking.walls_b,
    )
    assert isinstance(
        beta_idx_test_res1, float
    ) and 0.0 <= beta_idx_test_res1 <= np.sqrt(
        2 * (cfg1.behaviours.class_m.box_init + cfg1.behaviours.class_m.box_size) ** 2
    ), "Should be a positive float <= `diagonal box dims environment`."
    beta_idx_test_multi_res = []
    for swarm_config_positions in positions1:
        beta_idx_test_multi_res.append(
            compute_beta_index_measure(
                swarm_config_positions,
                cfg1.behaviours.flocking.box_size,
                cfg1.behaviours.flocking.walls_b,
            )
        )
    assert len(beta_idx_test_multi_res) == len(
        positions1
    ), "Should have the same shape."


def test_center_of_mass_metric(positions1: np.array, cfg1: OmegaConf):
    """"""
    c_o_m_test_res1 = compute_center_of_mass_swarm_configuration(positions1[0])
    assert (
        isinstance(c_o_m_test_res1, tuple)
        and 0.0
        <= c_o_m_test_res1[0]
        <= (cfg1.behaviours.class_m.box_init + cfg1.behaviours.class_m.box_size)
        and 0.0
        <= c_o_m_test_res1[1]
        <= (cfg1.behaviours.class_m.box_init + cfg1.behaviours.class_m.box_size)
    ), "Both values should be within [box_init, box_size]."
    c_o_m_test_multi_res = []
    for swarm_config_positions in positions1:
        c_o_m_test_multi_res.append(
            compute_center_of_mass_swarm_configuration(swarm_config_positions)
        )
    assert len(c_o_m_test_multi_res) == len(positions1), "Should have the same shape."


def test_maximum_swarm_shift_metric(positions1: np.array, cfg1: OmegaConf):
    """"""
    max_swarm_shift_test_res1 = compute_maximum_swarm_shift_swarm_configuration(
        positions1[0], cfg1.behaviours.class_m.walls_b, cfg1.behaviours.class_m.box_size
    )
    assert isinstance(
        max_swarm_shift_test_res1, float
    ) and 0.0 <= max_swarm_shift_test_res1 <= np.sqrt(
        2 * (cfg1.behaviours.class_m.box_init + cfg1.behaviours.class_m.box_size) ** 2
    ), "Should be a positive float <= `diagonal box dims environment`."
    max_swarm_shift_test_multi_res = []
    for swarm_config_positions in positions1:
        max_swarm_shift_test_multi_res.append(
            compute_maximum_swarm_shift_swarm_configuration(
                swarm_config_positions,
                cfg1.behaviours.class_m.walls_b,
                cfg1.behaviours.class_m.box_size,
            )
        )
    assert len(max_swarm_shift_test_multi_res) == len(
        positions1
    ), "Should have the same shape."


def test_swarm_mode_index_metric(
    positions1: np.array, cfg1: OmegaConf, curr_cfg: OmegaConf
):
    """"""
    swarm_mode_idx_test_res1 = compute_swarm_mode_index_swarm_configuration(
        positions1[0], curr_cfg.metrics.swarm_mode_index.freq_dist_threshold
    )
    assert isinstance(
        swarm_mode_idx_test_res1, float
    ) and 0.0 <= swarm_mode_idx_test_res1 <= np.sqrt(
        2 * (cfg1.behaviours.class_m.box_init + cfg1.behaviours.class_m.box_size) ** 2
    ), "Should be a positive float <= `diagonal box dims environment`."
    swarm_mode_idx_test_multi_res = []
    for swarm_config_positions in positions1:
        swarm_mode_idx_test_multi_res.append(
            compute_swarm_mode_index_swarm_configuration(
                swarm_config_positions,
                curr_cfg.metrics.swarm_mode_index.freq_dist_threshold,
            )
        )
    assert len(swarm_mode_idx_test_multi_res) == len(
        positions1
    ), "Should have the same shape."


def test_longest_path_metric(positions1: np.array, cfg1: OmegaConf):
    """"""
    longest_path_test_res1 = compute_longest_path_swarm_configuration(positions1[0])
    assert isinstance(
        longest_path_test_res1, float
    ) and 0.0 <= longest_path_test_res1 <= np.sqrt(
        2 * (cfg1.behaviours.class_m.box_init + cfg1.behaviours.class_m.box_size) ** 2
    ), "Should be a positive float <= `diagonal box dims environment`."
    longest_path_test_multi_res = []
    for swarm_config_positions in positions1:
        longest_path_test_multi_res.append(
            compute_longest_path_swarm_configuration(swarm_config_positions)
        )
    assert len(longest_path_test_multi_res) == len(
        positions1
    ), "Should have the same shape."


def test_maximum_radius_metric(positions1: np.array, cfg1: OmegaConf):
    """"""
    maximum_radius_test_res1 = compute_maximum_radius_swarm_configuration(positions1[0])
    assert isinstance(
        maximum_radius_test_res1, float
    ) and 0.0 <= maximum_radius_test_res1 <= np.sqrt(
        2 * (cfg1.behaviours.class_m.box_init + cfg1.behaviours.class_m.box_size) ** 2
    ), "Should be a positive float <= `diagonal box dims environment`."
    maximum_radius_test_multi_res = []
    for swarm_config_positions in positions1:
        maximum_radius_test_multi_res.append(
            compute_maximum_radius_swarm_configuration(swarm_config_positions)
        )
    assert len(maximum_radius_test_multi_res) == len(
        positions1
    ), "Should have the same shape."


def test_average_nearest_neighbour_distance_metric(
    positions1: np.ndarray, cfg1: OmegaConf
) -> None:
    avg_nearest_neigh_d_test_res1 = compute_average_nearest_neighbour_distance_measure(
        positions1[0]
    )
    assert (
        isinstance(avg_nearest_neigh_d_test_res1, float)
        and avg_nearest_neigh_d_test_res1 >= 0.0
    ), "Should be a positive float."
    avg_nearest_neigh_d_test_multi_res = []
    for swarm_config_positions in positions1:
        avg_nearest_neigh_d_test_multi_res.append(
            compute_average_nearest_neighbour_distance_measure(swarm_config_positions)
        )
    assert len(avg_nearest_neigh_d_test_multi_res) == len(
        positions1
    ), "Should have the same shape."


def test_collision_count_metric(positions1: np.ndarray, cfg1: OmegaConf) -> None:
    collision_c_test_res1 = compute_collision_count_measure(
        positions1[0],
        cfg1.behaviours.class_m.reynolds.alignm_dist_thres,
        cfg1.behaviours.class_m.walls_b,
        cfg1.behaviours.class_m.box_size,
    )
    assert (
        isinstance(collision_c_test_res1, int) and collision_c_test_res1 >= 0
    ), "Should be a positive integer."
    collision_c_test_multi_res = []
    for swarm_config_positions in positions1:
        collision_c_test_multi_res.append(
            compute_collision_count_measure(
                swarm_config_positions,
                cfg1.behaviours.class_m.reynolds.alignm_dist_thres,
                cfg1.behaviours.class_m.walls_b,
                cfg1.behaviours.class_m.box_size,
            )
        )
    assert len(collision_c_test_multi_res) == len(
        positions1
    ), "Should have the same shape."


def test_flock_density_metric(positions1: np.ndarray) -> None:
    flock_densi_test_res1 = compute_flock_density_measure(positions1[0])
    assert (
        isinstance(flock_densi_test_res1, float) and flock_densi_test_res1 >= 0.0
    ), "Should be a positive float."
    flock_densi_test_multi_res = []
    for swarm_config_positions in positions1:
        flock_densi_test_multi_res.append(
            compute_flock_density_measure(swarm_config_positions)
        )
    assert len(flock_densi_test_multi_res) == len(
        positions1
    ), "Should have the same shape."


def test_grouping_metric(positions1: np.ndarray, cfg1: OmegaConf) -> None:
    grouping_test_res1 = compute_grouping_measure(
        positions1[0], cfg1.behaviours.class_m.walls_b, cfg1.behaviours.class_m.box_size
    )
    assert (
        isinstance(grouping_test_res1, float) and grouping_test_res1 >= 0.0
    ), "Should be a positive float."
    grouping_test_multi_res = []
    for swarm_config_positions in positions1:
        grouping_test_multi_res.append(
            compute_grouping_measure(
                swarm_config_positions,
                cfg1.behaviours.class_m.walls_b,
                cfg1.behaviours.class_m.box_size,
            )
        )
    assert len(grouping_test_multi_res) == len(
        positions1
    ), "Should have the same shape."


def test_straggler_count_metric(positions1: np.ndarray, cfg1: OmegaConf) -> None:
    straggler_c_test_res1 = compute_straggler_count_measure(
        positions1[0],
        cfg1.behaviours.class_m.reynolds.alignm_dist_thres,
        cfg1.behaviours.class_m.walls_b,
        cfg1.behaviours.class_m.box_size,
    )
    assert (
        isinstance(straggler_c_test_res1, int) and straggler_c_test_res1 >= 0
    ), "Should be a positive integer."
    straggler_c_test_multi_res = []
    for swarm_config_positions in positions1:
        straggler_c_test_multi_res.append(
            compute_straggler_count_measure(
                swarm_config_positions,
                cfg1.behaviours.class_m.reynolds.alignm_dist_thres,
                cfg1.behaviours.class_m.walls_b,
                cfg1.behaviours.class_m.box_size,
            )
        )
    assert len(straggler_c_test_multi_res) == len(
        positions1
    ), "Should have the same shape."


def test_order_metric(velocities: np.ndarray) -> None:
    order_test_res1 = compute_order_measure(velocities[0])
    assert (
        isinstance(order_test_res1, tuple) and len(order_test_res1) == 2
    ), "Should be a np.array of shape (2,)."
    order_test_multi_res = []
    for swarm_config_vels in velocities:
        order_test_multi_res.append(compute_order_measure(swarm_config_vels))
    assert len(order_test_multi_res) == len(velocities), "Should have the same shape."


def test_subgroup_count_metric(positions1: np.ndarray, cfg1: OmegaConf) -> None:
    subgroup_c_test_res1 = compute_subgroup_count_measure(
        positions1[0],
        cfg1.behaviours.class_m.reynolds.alignm_dist_thres,
        cfg1.behaviours.class_m.walls_b,
        cfg1.behaviours.class_m.box_size,
    )
    assert (
        isinstance(subgroup_c_test_res1, int) and subgroup_c_test_res1 > 0
    ), "Should be a positive integer."
    subgroup_c_test_multi_res = []
    for swarm_config_positions in positions1:
        subgroup_c_test_multi_res.append(
            compute_subgroup_count_measure(
                swarm_config_positions,
                cfg1.behaviours.class_m.reynolds.alignm_dist_thres,
                cfg1.behaviours.class_m.walls_b,
                cfg1.behaviours.class_m.box_size,
            )
        )
    assert len(subgroup_c_test_multi_res) == len(
        positions1
    ), "Should have the same shape."


def test_diffusion_metric(positions1: np.ndarray, curr_cfg: OmegaConf) -> None:
    diffusion_test_res1 = compute_diffusion_measure(
        positions1, curr_cfg.metrics.diffusion.time_lag
    )
    assert isinstance(diffusion_test_res1[0], float) and isinstance(
        diffusion_test_res1[1], float
    ), "Should be a tuple of floats."


def test_neighbor_shortest_distances_metric(
    positions1: np.ndarray, curr_cfg: OmegaConf
) -> None:
    test_arena_radius = np.sqrt(
        2
        * (
            (
                curr_cfg.behaviours.class_m.box_init
                + curr_cfg.behaviours.class_m.box_size
            )
            // 2
        )
        ** 2
    )
    neig_s_d_test_res1 = compute_neighbor_shortest_distances_measure(
        positions1[0],
        test_arena_radius,
        curr_cfg.metrics.neighbor_shortest_distances.desc_sort,
        curr_cfg.behaviours.class_m.walls_b,
        curr_cfg.behaviours.class_m.box_size,
    )
    assert isinstance(neig_s_d_test_res1, np.ndarray) and len(
        neig_s_d_test_res1
    ) == len(positions1[0]), "Should be a numpy.array of `N` agents size."
    neig_s_d_test_multi_res = []
    for swarm_config_positions in positions1:
        neig_s_d_test_multi_res.append(
            compute_neighbor_shortest_distances_measure(
                swarm_config_positions,
                test_arena_radius,
                curr_cfg.metrics.neighbor_shortest_distances.desc_sort,
                curr_cfg.behaviours.class_m.walls_b,
                curr_cfg.behaviours.class_m.box_size,
            )
        )
    assert len(neig_s_d_test_multi_res) == len(
        positions1
    ), "Should have the same shape."


@hydra.main(config_path="../swarm_hydra/configs", config_name="config", version_base="1.1")
def main(cfg: DictConfig):
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    hydra_cfg["runtime"]["output_dir"]
    out_dir = hydra_cfg["runtime"]["output_dir"]

    # Redirect stdout & stderr
    sys.stdout = LoggerWriter(log, logging.INFO)
    sys.stderr = LoggerWriter(log, logging.ERROR)

    # Unit testing

    test_utils_metrics(experi_dir_flocking, cfg)

    cfg_experi_rey, hdf5_experi_rey, reynolds_subfolders = processing_folder_structure(
        experi_dir_flocking, "reynolds"
    )
    cfg_experi_vic, hdf5_experi_vic, vicsek_subfolders = processing_folder_structure(
        experi_dir_flocking, "vicsek"
    )

    test_mse_metric(
        hdf5_experi_rey[reynolds_subfolders[0]].get("boids_pos"),
        cfg,
    )

    """
    # Testing K-tree application for Boid tasks
    k_tree_edges, spatial_weights = construct_boids_k_tree(
        Boids_Reynolds(X=boids_pos_arr[-1], X_dot=boids_vel_arr[-1]),
        cfg.behaviours.class_m.reynolds.force,
        {"box_size": cfg.behaviours.class_m.box_size,
        "dt": cfg.behaviours.class_m.dt,
        "inner_loop": cfg.behaviours.class_m.inner_loop,
        "sim_speed": cfg.behaviours.class_m.sim_speed,
        "max_speed": cfg.behaviours.class_m.max_abs_sp,
        "coef_alignment": cfg.behaviours.class_m.reynolds.alignm_w,
        "d_align": cfg.behaviours.class_m.reynolds.alignm_dist_thres,
        "coef_cohesion": cfg.behaviours.class_m.reynolds.cohe_w,
        "d_cohe": cfg.behaviours.class_m.reynolds.cohe_dist_thres,
        "coef_separation": cfg.behaviours.class_m.reynolds.separ_w,
        "d_sepa": cfg.behaviours.class_m.reynolds.separ_dist_thres,
        "f_lim": cfg.behaviours.class_m.max_abs_f,
        "v_lim": cfg.behaviours.class_m.max_abs_sp,
        "v_const": cfg.behaviours.class_m.init_speed,
        "eps": cfg.behaviours.class_m.eps,
        "walls_b": cfg.behaviours.class_m.walls_b,
        "store_interm_data":cfg.behaviours.class_m.storing_data}
    )
    assert len(k_tree_edges) > 0, "The k-tree should have edges."
    assert len(spatial_weights) > 0, "The k-tree should have weighted edges."
    plot_kinematic_tree_and_swarm(
        k_tree_edges,
        spatial_weights,
        boids_pos_arr[-1],
        'dummy_k_tree.png'
    )
    """

    test_area_metric(
        hdf5_experi_rey[reynolds_subfolders[0]].get("boids_pos"),
        cfg_experi_rey[reynolds_subfolders[0]],
    )

    test_polarisation_metric(hdf5_experi_rey[reynolds_subfolders[0]].get("boids_vels"))

    test_combined_state_count_metric(
        hdf5_experi_rey[reynolds_subfolders[0]].get("boids_pos"),
        hdf5_experi_vic[vicsek_subfolders[0]].get("boids_pos"),
        hdf5_experi_rey[reynolds_subfolders[0]].get("boids_vels"),
        hdf5_experi_vic[vicsek_subfolders[0]].get("boids_vels"),
        cfg,
    )

    test_sampled_average_state_metric(
        hdf5_experi_rey[reynolds_subfolders[0]].get("boids_pos"),
        hdf5_experi_vic[vicsek_subfolders[0]].get("boids_pos"),
        hdf5_experi_rey[reynolds_subfolders[0]].get("boids_vels"),
        hdf5_experi_vic[vicsek_subfolders[0]].get("boids_vels"),
        cfg,
    )

    test_average_local_density_metric(
        hdf5_experi_rey[reynolds_subfolders[0]].get("boids_pos"),
        cfg_experi_rey[reynolds_subfolders[0]],
    )

    test_beta_index_metric(
        hdf5_experi_rey[reynolds_subfolders[0]].get("boids_pos"),
        cfg_experi_vic[vicsek_subfolders[0]],
    )

    test_center_of_mass_metric(
        hdf5_experi_rey[reynolds_subfolders[0]].get("boids_pos"),
        cfg_experi_rey[reynolds_subfolders[0]],
    )

    test_maximum_swarm_shift_metric(
        hdf5_experi_rey[reynolds_subfolders[0]].get("boids_pos"),
        cfg_experi_rey[reynolds_subfolders[0]],
    )

    test_swarm_mode_index_metric(
        hdf5_experi_rey[reynolds_subfolders[0]].get("boids_pos"),
        cfg_experi_rey[reynolds_subfolders[0]],
        cfg,
    )

    test_longest_path_metric(
        hdf5_experi_rey[reynolds_subfolders[0]].get("boids_pos"),
        cfg_experi_rey[reynolds_subfolders[0]],
    )

    test_maximum_radius_metric(
        hdf5_experi_rey[reynolds_subfolders[0]].get("boids_pos"),
        cfg_experi_rey[reynolds_subfolders[0]],
    )

    test_average_nearest_neighbour_distance_metric(
        hdf5_experi_rey[reynolds_subfolders[0]].get("boids_pos"),
        cfg_experi_rey[reynolds_subfolders[0]],
    )

    test_collision_count_metric(
        hdf5_experi_rey[reynolds_subfolders[0]].get("boids_pos"),
        cfg_experi_rey[reynolds_subfolders[0]],
    )

    test_flock_density_metric(
        hdf5_experi_rey[reynolds_subfolders[0]].get("boids_pos"),
    )

    test_grouping_metric(
        hdf5_experi_rey[reynolds_subfolders[0]].get("boids_pos"),
        cfg_experi_rey[reynolds_subfolders[0]],
    )

    test_straggler_count_metric(
        hdf5_experi_rey[reynolds_subfolders[0]].get("boids_pos"),
        cfg_experi_rey[reynolds_subfolders[0]],
    )

    test_order_metric(
        hdf5_experi_rey[reynolds_subfolders[0]].get("boids_vels"),
    )

    test_subgroup_count_metric(
        hdf5_experi_rey[reynolds_subfolders[0]].get("boids_pos"),
        cfg_experi_rey[reynolds_subfolders[0]],
    )

    test_diffusion_metric(hdf5_experi_rey[reynolds_subfolders[0]].get("boids_pos"), cfg)

    test_neighbor_shortest_distances_metric(
        hdf5_experi_rey[reynolds_subfolders[0]].get("boids_pos"), cfg
    )


if __name__ == "__main__":
    main()
