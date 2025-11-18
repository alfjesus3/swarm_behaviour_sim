import hydra
from omegaconf import DictConfig, OmegaConf
import logging
import numpy as np
import sys
import os
from typing import Dict, List, Any, Optional, Callable
from enum import Enum

import matplotlib.pyplot as plt
import matplotlib
import itertools
import h5py
import gc
from collections import defaultdict, Counter
from itertools import combinations
import scipy

from swarm_hydra.entry_point import *
from swarm_hydra.metrics.utils_metrics import processing_folder_structure
from swarm_hydra.metrics.spatial_metrics import (
    compute_cosine_similarity_measure,
    compute_combined_state_count_measure,
    compute_sampled_average_state_measure,
    compute_euclidean_distance_measure,
)
from swarm_hydra.metrics.swarm_features_combos import get_metrics_combo, FeaturesCombos
from swarm_hydra.metrics.utils_metrics import (
    open_dataframe,
    append_row_to_dataframe,
    export_dataframe,
)

from statsmodels.tsa.api import SimpleExpSmoothing

# A logger for this file
log = logging.getLogger(__name__)


def _export_behaviour_features_to_npz(
    behaviour_swarm_metrics: Dict[str, np.ndarray],
    behaviour_features_path: str,
    subfolder_names: List[str],
    target_swarm_metrics: List[str],
    range_steps: range,
    num_boids: int,
) -> None:
    """
    Export behaviour_swarm_metrics to NPZ format with metadata.

    Args:
        behaviour_swarm_metrics: Dictionary of computed behaviour metrics
        behaviour_features_path: File path to save the NPZ file
        subfolder_names: List of subfolder names processed
        target_swarm_metrics: List of target swarm metrics computed
        range_steps: Range of steps that were processed
        num_boids: Number of boids in the simulation
    """
    # Prepare data dictionary for NPZ export
    npz_data = {}

    # Add the main behaviour metrics data
    for metric_name, metric_data in behaviour_swarm_metrics.items():
        npz_data[f"metric_{metric_name}"] = metric_data

    # Add metadata for reconstruction and reference
    npz_data.update(
        {
            "subfolder_names": np.array(subfolder_names, dtype=object),
            "target_swarm_metrics": np.array(target_swarm_metrics, dtype=object),
            "range_steps_start": np.array([range_steps.start]),
            "range_steps_stop": np.array([range_steps.stop]),
            "range_steps_step": np.array([range_steps.step]),
            "num_boids": np.array([num_boids]),
            "n_subfolders": np.array([len(subfolder_names)]),
            "n_steps": np.array([range_steps.stop - range_steps.start]),
            "n_metrics": np.array([len(target_swarm_metrics)]),
            "data_shape": np.array(
                [len(subfolder_names), range_steps.stop - range_steps.start]
            ),
        }
    )

    # Save as compressed NPZ file
    np.savez_compressed(behaviour_features_path, **npz_data)
    print(f"Behaviour features exported to: {behaviour_features_path}")


def load_behaviour_features_from_npz(behaviour_features_path: str) -> Dict[str, Any]:
    """
    Load behaviour features from NPZ file.

    Args:
        behaviour_features_path: Path to the NPZ file containing behaviour features

    Returns:
        Dictionary containing:
        - 'behaviour_swarm_metrics': The main metrics data (same format as compute_behaviour_features output)
        - 'metadata': Dictionary with reconstruction information
    """
    if not os.path.exists(behaviour_features_path):
        raise FileNotFoundError(
            f"Behaviour features file not found: {behaviour_features_path}"
        )

    # Load NPZ file
    npz_data = np.load(behaviour_features_path, allow_pickle=True)

    # Extract metadata
    metadata = {
        "subfolder_names": list(npz_data["subfolder_names"]),
        "target_swarm_metrics": list(npz_data["target_swarm_metrics"]),
        "range_steps": range(
            npz_data["range_steps_start"][0],
            npz_data["range_steps_stop"][0],
            npz_data["range_steps_step"][0],
        ),
        "num_boids": npz_data["num_boids"][0],
        "n_subfolders": npz_data["n_subfolders"][0],
        "n_steps": npz_data["n_steps"][0],
        "n_metrics": npz_data["n_metrics"][0],
        "data_shape": tuple(npz_data["data_shape"]),
    }

    # Reconstruct behaviour_swarm_metrics dictionary
    behaviour_swarm_metrics = {}
    for metric_name in metadata["target_swarm_metrics"]:
        behaviour_swarm_metrics[metric_name] = npz_data[f"metric_{metric_name}"]

    # Close NPZ file
    npz_data.close()

    return {"behaviour_swarm_metrics": behaviour_swarm_metrics, "metadata": metadata}


def compute_behaviour_features(
    subfolder_names,
    experi_cfgs,
    experi_hdf5s,
    swarm_feats,
    num_boids,
    range_steps,
    behaviour_features_path: Optional[str] = None,
) -> dict:
    """"""
    n_steps = range_steps.stop - range_steps.start
    n_metrics = len(swarm_feats.names)
    n_subfolders = len(subfolder_names)

    # Pre-allocate final array with known dimensions
    n_agents = swarm_feats.n_agents
    all_metrics_data = np.empty(
        (n_subfolders, n_steps, n_agents, n_metrics), dtype=object
    )  # Shape: (num_simulations, T, num_robots, num_features)

    # Convert range to list once for indexing
    steps_list = list(range_steps)

    for subfolder_idx, behaviour_subf in enumerate(subfolder_names):
        behaviour_subf_cfg = experi_cfgs[behaviour_subf]
        behaviour_subf_hdf5 = experi_hdf5s[behaviour_subf]

        # Cache frequently accessed attributes
        interm_data_names = (
            behaviour_subf_cfg.behaviours.class_m.model.interm_data_names
        )
        boids_pos_idx = interm_data_names[1]
        boids_vels_idx = interm_data_names[2]

        # Cache max_dist calculation
        model_cfg = behaviour_subf_cfg.behaviours.class_m.model
        max_dist = (
            model_cfg.max_dist
            if hasattr(model_cfg, "max_dist")
            else model_cfg.alignm_dist_thres
        )

        # Pre-fetch HDF5 data to avoid repeated access
        boids_pos_data = behaviour_subf_hdf5.get(boids_pos_idx)
        boids_vels_data = behaviour_subf_hdf5.get(boids_vels_idx)

        # Vectorized computation over steps
        for step_idx, step in enumerate(steps_list):
            metrics_result = get_metrics_combo(
                swarm_feats,
                cfg=behaviour_subf_cfg,
                boids_pos=boids_pos_data[step],
                boids_vels=boids_vels_data[step],
                radius_sensing=max_dist,
                store_interm_data=False,
            )
            all_metrics_data[subfolder_idx, step_idx, :, :] = metrics_result

    # Reshape and create final dictionary
    # Shape: (num_simulations, T, 1, num_features) -> num_features * (num_simulations, T, 1, 1)
    reshaped_data = np.split(all_metrics_data, all_metrics_data.shape[-1], -1)

    behaviour_swarm_metrics = {
        swarm_feats.names[i]: reshaped_data[i] for i in range(n_metrics)
    }

    # Export to NPZ
    _export_behaviour_features_to_npz(
        behaviour_swarm_metrics,
        behaviour_features_path,
        subfolder_names,
        swarm_feats.names,
        range_steps,
        num_boids,
    )

    return behaviour_swarm_metrics


def minmax_normalize_along_axis1(arr, eps=1e-9):
    """Normalizing 3d np.array in the 2nd dimensions"""
    min_vals = np.min(arr, axis=1, keepdims=True)
    max_vals = np.max(arr, axis=1, keepdims=True)
    if isinstance(min_vals.flatten()[0], tuple) and isinstance(
        max_vals.flatten()[0], tuple
    ):
        norms = np.linalg.norm(np.array(arr[:, :, :, 0].tolist()), axis=-1)
        norms = np.where(norms == 0, eps, norms)
        arr_flatten = np.array(arr[:, :, :, 0].tolist())
        out_vals = arr_flatten / norms[:, :, :, np.newaxis]
    else:
        range_vals = max_vals - min_vals + eps  # avoid division by zero
        out_vals = (arr - min_vals) / range_vals
    return out_vals.astype(float)


def visualize_behaviour_measures(
    all_run_idxs: np.ndarray,
    num_samples_run: int,
    behaviour_data: dict,
    behaviours_swarm_metrics_lst: list,
    colors: dict,
    file_name: str,
) -> None:
    """"""
    for i, (run_idx) in enumerate(all_run_idxs):
        groups = {}
        for k, v in behaviour_data.items():
            groups[k] = {"x": np.arange(num_samples_run)}
            for y_level in range(len(v)):
                groups[k][f"y{y_level}"] = v[behaviours_swarm_metrics_lst[y_level]][
                    run_idx
                ]

        # # Apply SimpleExpSmoothing to all y{} entries in the groups dictionary.
        # for group_name, group_data in groups.items():
        #     for key, value in group_data.items():
        #         if key.startswith("y"):
        #             time_series = np.array(value).flatten()
        #             if len(time_series) > 1:
        #                 smoother = SimpleExpSmoothing(time_series)
        #                 groups[group_name][key] = smoother.fit(
        #                     smoothing_level=0.05, optimized=False
        #                 ).fittedvalues

        # Create subplot titles
        subplot_titles = behaviours_swarm_metrics_lst

        # Create figure with N subplots
        num_rows, num_cols = (
            len(behaviours_swarm_metrics_lst) // 2 + 1,
            len(behaviours_swarm_metrics_lst) // 2,
        )
        fig, axes = plt.subplots(
            num_rows,
            num_cols,
            figsize=(12, 15),
            sharey=True,
            sharex=True,
        )

        # Set overall figure title
        fig.suptitle(
            f"Swarm measures for run {run_idx} in the training dataset",
            fontsize=16,
            fontweight="bold",
        )

        # Plot data on each subplot (flatten axes for easy iteration)
        axes_flat = axes.flatten()

        for i in range(len(behaviours_swarm_metrics_lst)):
            ax = axes_flat[i]

            # Plot each group with consistent colors
            for group_name, data in groups.items():
                if data[f"y{i}"].shape[-1] != 1:
                    continue  # Not plotting multi-dim variables like `center_of_mass`
                # Add some variation to each subplot (optional)
                x_variation = np.repeat(data["x"], data[f"y{i}"].shape[1])
                y_variation = data[f"y{i}"].flatten()

                ax.scatter(
                    x_variation,
                    y_variation,
                    color=colors[group_name],
                    alpha=0.7,
                    s=10,
                    label=group_name,
                )

            # Customize each subplot
            ax.set_title(subplot_titles[i], fontsize=12, fontweight="bold")
            ax.grid(True, alpha=0.3)

            # Add legend only to the first subplot to avoid clutter
            if i == 0:
                ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

        # Hide the empty subplots
        for j in range(len(behaviours_swarm_metrics_lst), len(axes_flat)):
            axes_flat[j].set_visible(False)

        # Set consistent axis limits for all subplots
        x_min, x_max = (-1, num_samples_run)
        y_min, y_max = -0.1, 1.1

        for i in range(num_rows):
            for j in range(num_cols):
                axes[i, j].set_xlabel("X Values")
                axes[i, j].set_ylabel("Y Values")
                axes[i, j].set_xlim(x_min, x_max)
                axes[i, j].set_ylim(y_min, y_max)

        # Adjust layout to prevent overlap
        plt.tight_layout()

        # # Show the plot
        # plt.show()
        # Alternative: Save the figure
        plt.savefig(
            file_name.format(run_idx),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close("all")
        gc.collect()  # Force garbage collection after each batch


def process_swarm_metrics_independently(
    swarm_metrics_dict,
    metric_keys,
    indices,
    num_sliding_w: int,
    num_overlap: int = 0,
    applying_smoothing: bool = True,
    smoothing_level: float = 0.3,
):
    """
    Process swarm metrics with independent smoothing for each index and sliding window support.

    This function clearly shows that each swarm_metrics[m][indices[i]]
    is smoothed independently from swarm_metrics[m][indices[j]]

    Args:
        swarm_metrics_dict: Dictionary containing metric data
        metric_keys: Keys for metrics to process
        indices: Indices to process for each metric
        num_sliding_w: Size of each sliding window
        num_overlap: Number of overlapping elements between consecutive windows (0 to num_sliding_w-1)
        applying_smoothing: Whether to apply exponential smoothing
        smoothing_level: Smoothing parameter for exponential smoothing

    Returns:
        Concatenated array of processed metrics with sliding windows
    """
    # Validate num_overlap parameter
    if not (0 <= num_overlap < num_sliding_w):
        raise ValueError(
            f"num_overlap must be between 0 and {num_sliding_w-1}, got {num_overlap}"
        )

    processed_metrics = []

    for m in metric_keys:
        # Collect smoothed series for this metric
        series_list = []

        for idx in indices:
            # Get individual time series: swarm_metrics[m][idx]
            individual_series = swarm_metrics_dict[m][idx]

            # Apply smoothing to this specific series independently
            if applying_smoothing:
                try:
                    from statsmodels.tsa.holtwinters import SimpleExpSmoothing

                    smoother = SimpleExpSmoothing(individual_series)
                    smoothed_series = smoother.fit(
                        smoothing_level=smoothing_level, optimized=False
                    ).fittedvalues
                    series_list.append(smoothed_series)
                except:
                    series_list.append(individual_series)
            else:
                series_list.append(individual_series)

        # Stack the independently smoothed series
        metric_data = np.array(series_list)  # Shape: (len(indices), time_steps)

        # Apply sliding window with overlap to each series
        windowed_data = []
        for series_idx in range(len(indices)):
            series = metric_data[series_idx]

            # Create sliding windows with overlap
            step_size = num_sliding_w - num_overlap
            windows = []

            start = 0
            while start + num_sliding_w <= len(series):
                window = series[start : start + num_sliding_w]
                windows.append(window)
                start += step_size

            if windows:  # Only add if we have at least one complete window
                windowed_data.append(np.array(windows))

        if windowed_data:  # Only add if we have data
            # Stack all windowed series for this metric
            # Shape: (len(indices), num_windows, num_sliding_w)
            metric_windowed = np.array(windowed_data)
            processed_metrics.append(
                metric_windowed.reshape(
                    metric_windowed.shape[0],
                    metric_windowed.shape[1],
                    metric_windowed.shape[2] * metric_windowed.shape[3],
                    metric_windowed.shape[4],
                )
            )

    if not processed_metrics:
        return np.array([])

    # Concatenate along the last dimension
    return np.concatenate(processed_metrics, axis=-1)


def process_all_behaviours_variants(
    swarm_feats: Enum,
    range_generate_and_optimize: range,
    load_npz_paths: list,
    curr_cfg: DictConfig,
) -> tuple:
    """"""
    ######
    # Preparing `40b` data for Reynolds, Vicsek, Aggregation, Dispersion, Brownian and Ballistic Motion Random Walk
    ######
    if os.path.exists(load_npz_paths[0]):
        rey_swarm_metrics_40b = load_behaviour_features_from_npz(load_npz_paths[0])[
            "behaviour_swarm_metrics"
        ]
        for k, v in rey_swarm_metrics_40b.items():
            rey_swarm_metrics_40b[k] = v[
                :, range_generate_and_optimize.start : range_generate_and_optimize.stop
            ]
        num_rey_subfolders = list(rey_swarm_metrics_40b.values())[0].shape[0]
        rey_all_f_idxs = np.array([i for i in range(num_rey_subfolders)])
    else:
        cfg_experi_rey, hdf5_experi_rey, reynolds_subfolders = (
            processing_folder_structure(
                "/home/aj/Documents/repos/outputs/2025-06-05/21-12-52/", "reynolds"
            )
        )
        # TODO temporary since these config parts do not exist in earlier datalogs ...
        new_swarm_feats = None
        for _, temp_cfg in cfg_experi_rey.items():
            temp_cfg.metrics.swarm_mode_index = curr_cfg.metrics.swarm_mode_index
            temp_cfg.metrics.diffusion = curr_cfg.metrics.diffusion
            temp_cfg.metrics.neighbor_shortest_distances = (
                curr_cfg.metrics.neighbor_shortest_distances
            )
            temp_cfg.features = curr_cfg.features
            temp_cfg.features.LocalBoidsFeats.n_agents = (
                temp_cfg.behaviours.class_m.boid_count
            )
            temp_cfg.features.Kuckling2023.n_agents = (
                temp_cfg.behaviours.class_m.boid_count
            )
            if new_swarm_feats is None:
                new_swarm_feats = hydra.utils.instantiate(
                    temp_cfg.features
                ).get_selected()
        rey_all_f_idxs = np.array(
            cfg_experi_rey[reynolds_subfolders[0]].seeds_per_exp_rep
        )
        num_rey_subfolders = len(
            cfg_experi_rey[reynolds_subfolders[0]].seeds_per_exp_rep
        )
        rey_swarm_metrics_40b = compute_behaviour_features(
            subfolder_names=reynolds_subfolders,
            experi_cfgs=cfg_experi_rey,
            experi_hdf5s=hdf5_experi_rey,
            swarm_feats=new_swarm_feats,
            num_boids=hdf5_experi_rey[reynolds_subfolders[0]].get("boids_pos").shape[1],
            range_steps=range_generate_and_optimize,
            behaviour_features_path="rey_swarm_metrics_40b.npz",
        )
        del cfg_experi_rey, hdf5_experi_rey, reynolds_subfolders

    if os.path.exists(load_npz_paths[1]):
        vic_swarm_metrics_40b = load_behaviour_features_from_npz(load_npz_paths[1])[
            "behaviour_swarm_metrics"
        ]
        for k, v in vic_swarm_metrics_40b.items():
            vic_swarm_metrics_40b[k] = v[
                :, range_generate_and_optimize.start : range_generate_and_optimize.stop
            ]
    else:
        cfg_experi_vic, hdf5_experi_vic, vicsek_subfolders = (
            processing_folder_structure(
                "/home/aj/Documents/repos/outputs/2025-06-05/21-12-52/", "vicsek"
            )
        )
        # TODO temporary since these config parts do not exist in earlier datalogs ...
        new_swarm_feats = None
        for _, temp_cfg in cfg_experi_vic.items():
            temp_cfg.metrics.swarm_mode_index = curr_cfg.metrics.swarm_mode_index
            temp_cfg.metrics.diffusion = curr_cfg.metrics.diffusion
            temp_cfg.metrics.neighbor_shortest_distances = (
                curr_cfg.metrics.neighbor_shortest_distances
            )
            temp_cfg.features = curr_cfg.features
            temp_cfg.features.LocalBoidsFeats.n_agents = (
                temp_cfg.behaviours.class_m.boid_count
            )
            temp_cfg.features.Kuckling2023.n_agents = (
                temp_cfg.behaviours.class_m.boid_count
            )
            if new_swarm_feats is None:
                new_swarm_feats = hydra.utils.instantiate(
                    temp_cfg.features
                ).get_selected()
        vic_swarm_metrics_40b = compute_behaviour_features(
            subfolder_names=vicsek_subfolders,
            experi_cfgs=cfg_experi_vic,
            experi_hdf5s=hdf5_experi_vic,
            swarm_feats=new_swarm_feats,
            num_boids=hdf5_experi_vic[vicsek_subfolders[0]].get("boids_pos").shape[1],
            range_steps=range_generate_and_optimize,
            behaviour_features_path="vic_swarm_metrics_40b.npz",
        )
        del cfg_experi_vic, hdf5_experi_vic, vicsek_subfolders

    if os.path.exists(load_npz_paths[2]):
        aggreg_swarm_metrics_40b = load_behaviour_features_from_npz(load_npz_paths[2])[
            "behaviour_swarm_metrics"
        ]
        for k, v in aggreg_swarm_metrics_40b.items():
            aggreg_swarm_metrics_40b[k] = v[
                :, range_generate_and_optimize.start : range_generate_and_optimize.stop
            ]
    else:
        cfg_experi_aggreg, hdf5_experi_aggreg, aggregation_subfolders = (
            processing_folder_structure(
                "/home/aj/Documents/repos/outputs/2025-06-06/05-04-59/",
                "aggregation",
            )
        )
        # TODO temporary since these config parts do not exist in earlier datalogs ...
        new_swarm_feats = None
        for _, temp_cfg in cfg_experi_aggreg.items():
            temp_cfg.metrics.swarm_mode_index = curr_cfg.metrics.swarm_mode_index
            temp_cfg.metrics.diffusion = curr_cfg.metrics.diffusion
            temp_cfg.metrics.neighbor_shortest_distances = (
                curr_cfg.metrics.neighbor_shortest_distances
            )
            temp_cfg.features = curr_cfg.features
            temp_cfg.features.LocalBoidsFeats.n_agents = (
                temp_cfg.behaviours.class_m.boid_count
            )
            temp_cfg.features.Kuckling2023.n_agents = (
                temp_cfg.behaviours.class_m.boid_count
            )
            if new_swarm_feats is None:
                new_swarm_feats = hydra.utils.instantiate(
                    temp_cfg.features
                ).get_selected()
        aggreg_swarm_metrics_40b = compute_behaviour_features(
            subfolder_names=aggregation_subfolders,
            experi_cfgs=cfg_experi_aggreg,
            experi_hdf5s=hdf5_experi_aggreg,
            swarm_feats=new_swarm_feats,
            num_boids=hdf5_experi_aggreg[aggregation_subfolders[0]]
            .get("boids_pos")
            .shape[1],
            range_steps=range_generate_and_optimize,
            behaviour_features_path="aggreg_swarm_metrics_40b.npz",
        )
        del cfg_experi_aggreg, hdf5_experi_aggreg, aggregation_subfolders

    if os.path.exists(load_npz_paths[3]):
        disper_swarm_metrics_40b = load_behaviour_features_from_npz(load_npz_paths[3])[
            "behaviour_swarm_metrics"
        ]
        for k, v in disper_swarm_metrics_40b.items():
            disper_swarm_metrics_40b[k] = v[
                :, range_generate_and_optimize.start : range_generate_and_optimize.stop
            ]
    else:
        cfg_experi_disper, hdf5_experi_disper, dispersion_subfolders = (
            processing_folder_structure(
                "/home/aj/Documents/repos/outputs/2025-06-06/08-57-06/", "dispersion"
            )
        )
        # TODO temporary since these config parts do not exist in earlier datalogs ...
        new_swarm_feats = None
        for _, temp_cfg in cfg_experi_disper.items():
            temp_cfg.metrics.swarm_mode_index = curr_cfg.metrics.swarm_mode_index
            temp_cfg.metrics.diffusion = curr_cfg.metrics.diffusion
            temp_cfg.metrics.neighbor_shortest_distances = (
                curr_cfg.metrics.neighbor_shortest_distances
            )
            temp_cfg.features = curr_cfg.features
            temp_cfg.features.LocalBoidsFeats.n_agents = (
                temp_cfg.behaviours.class_m.boid_count
            )
            temp_cfg.features.Kuckling2023.n_agents = (
                temp_cfg.behaviours.class_m.boid_count
            )
            if new_swarm_feats is None:
                new_swarm_feats = hydra.utils.instantiate(
                    temp_cfg.features
                ).get_selected()
        disper_swarm_metrics_40b = compute_behaviour_features(
            subfolder_names=dispersion_subfolders,
            experi_cfgs=cfg_experi_disper,
            experi_hdf5s=hdf5_experi_disper,
            swarm_feats=new_swarm_feats,
            num_boids=hdf5_experi_disper[dispersion_subfolders[0]]
            .get("boids_pos")
            .shape[1],
            range_steps=range_generate_and_optimize,
            behaviour_features_path="disper_swarm_metrics_40b.npz",
        )
        del cfg_experi_disper, hdf5_experi_disper, dispersion_subfolders

    if os.path.exists(load_npz_paths[4]):
        balli_swarm_metrics_40b = load_behaviour_features_from_npz(load_npz_paths[4])[
            "behaviour_swarm_metrics"
        ]
        for k, v in balli_swarm_metrics_40b.items():
            balli_swarm_metrics_40b[k] = v[
                :, range_generate_and_optimize.start : range_generate_and_optimize.stop
            ]
    else:
        cfg_experi_balli, hdf5_experi_balli, ballistic_subfolders = (
            processing_folder_structure(
                "/home/aj/Documents/repos/outputs/2025-06-11/22-53-13/",
                "ballistic",
            )
        )
        # TODO temporary since `max_dist` is used in a different context on `ballistic` and `brownian`
        new_swarm_feats = None
        for _, temp_cfg in cfg_experi_balli.items():
            temp_cfg.metrics.swarm_mode_index = curr_cfg.metrics.swarm_mode_index
            temp_cfg.metrics.diffusion = curr_cfg.metrics.diffusion
            temp_cfg.metrics.neighbor_shortest_distances = (
                curr_cfg.metrics.neighbor_shortest_distances
            )
            temp_cfg.features = curr_cfg.features
            temp_cfg.features.LocalBoidsFeats.n_agents = (
                temp_cfg.behaviours.class_m.boid_count
            )
            temp_cfg.features.Kuckling2023.n_agents = (
                temp_cfg.behaviours.class_m.boid_count
            )
            if new_swarm_feats is None:
                new_swarm_feats = hydra.utils.instantiate(
                    temp_cfg.features
                ).get_selected()
            temp_cfg.behaviours.class_m.model.max_dist = 100
        balli_swarm_metrics_40b = compute_behaviour_features(
            subfolder_names=ballistic_subfolders,
            experi_cfgs=cfg_experi_balli,
            experi_hdf5s=hdf5_experi_balli,
            swarm_feats=new_swarm_feats,
            num_boids=hdf5_experi_balli[ballistic_subfolders[0]]
            .get("boids_pos")
            .shape[1],
            range_steps=range_generate_and_optimize,
            behaviour_features_path="balli_swarm_metrics_40b.npz",
        )
        del cfg_experi_balli, hdf5_experi_balli, ballistic_subfolders

    if os.path.exists(load_npz_paths[5]):
        brown_swarm_metrics_40b = load_behaviour_features_from_npz(load_npz_paths[5])[
            "behaviour_swarm_metrics"
        ]
        for k, v in brown_swarm_metrics_40b.items():
            brown_swarm_metrics_40b[k] = v[
                :, range_generate_and_optimize.start : range_generate_and_optimize.stop
            ]
    else:
        cfg_experi_brown, hdf5_experi_brown, brownian_subfolders = (
            processing_folder_structure(
                "/home/aj/Documents/repos/outputs/2025-06-11/22-53-13/",
                "brownian",
            )
        )
        # TODO temporary since `max_dist` is used in a different context on `ballistic` and `brownian`
        new_swarm_feats = None
        for _, temp_cfg in cfg_experi_brown.items():
            temp_cfg.metrics.swarm_mode_index = curr_cfg.metrics.swarm_mode_index
            temp_cfg.metrics.diffusion = curr_cfg.metrics.diffusion
            temp_cfg.metrics.neighbor_shortest_distances = (
                curr_cfg.metrics.neighbor_shortest_distances
            )
            temp_cfg.features = curr_cfg.features
            temp_cfg.features.LocalBoidsFeats.n_agents = (
                temp_cfg.behaviours.class_m.boid_count
            )
            temp_cfg.features.Kuckling2023.n_agents = (
                temp_cfg.behaviours.class_m.boid_count
            )
            if new_swarm_feats is None:
                new_swarm_feats = hydra.utils.instantiate(
                    temp_cfg.features
                ).get_selected()
            temp_cfg.behaviours.class_m.model.max_dist = 100
        brown_swarm_metrics_40b = compute_behaviour_features(
            subfolder_names=brownian_subfolders,
            experi_cfgs=cfg_experi_brown,
            experi_hdf5s=hdf5_experi_brown,
            swarm_feats=new_swarm_feats,
            num_boids=hdf5_experi_brown[brownian_subfolders[0]]
            .get("boids_pos")
            .shape[1],
            range_steps=range_generate_and_optimize,
            behaviour_features_path="brown_swarm_metrics_40b.npz",
        )
        del cfg_experi_brown, hdf5_experi_brown, brownian_subfolders

    ######
    # Preparing `30b` data for Reynolds, Vicsek, Aggregation, Dispersion, Brownian and Ballistic Motion Random Walk
    ######
    if os.path.exists(load_npz_paths[6]):
        rey_swarm_metrics_30b = load_behaviour_features_from_npz(load_npz_paths[6])[
            "behaviour_swarm_metrics"
        ]
        for k, v in rey_swarm_metrics_30b.items():
            rey_swarm_metrics_30b[k] = v[
                :, range_generate_and_optimize.start : range_generate_and_optimize.stop
            ]
        num_rey_subfolders = list(rey_swarm_metrics_30b.values())[0].shape[0]
    else:
        cfg_experi_rey, hdf5_experi_rey, reynolds_subfolders = (
            processing_folder_structure(
                "/home/aj/Documents/repos/outputs/2025-06-06/02-59-01/", "reynolds"
            )
        )
        # TODO temporary since these config parts do not exist in earlier datalogs ...
        new_swarm_feats = None
        for _, temp_cfg in cfg_experi_rey.items():
            temp_cfg.metrics.swarm_mode_index = curr_cfg.metrics.swarm_mode_index
            temp_cfg.metrics.diffusion = curr_cfg.metrics.diffusion
            temp_cfg.metrics.neighbor_shortest_distances = (
                curr_cfg.metrics.neighbor_shortest_distances
            )
            temp_cfg.features = curr_cfg.features
            temp_cfg.features.LocalBoidsFeats.n_agents = (
                temp_cfg.behaviours.class_m.boid_count
            )
            temp_cfg.features.Kuckling2023.n_agents = (
                temp_cfg.behaviours.class_m.boid_count
            )
            if new_swarm_feats is None:
                new_swarm_feats = hydra.utils.instantiate(
                    temp_cfg.features
                ).get_selected()
        num_rey_subfolders = len(
            cfg_experi_rey[reynolds_subfolders[0]].seeds_per_exp_rep
        )
        rey_swarm_metrics_30b = compute_behaviour_features(
            subfolder_names=reynolds_subfolders,
            experi_cfgs=cfg_experi_rey,
            experi_hdf5s=hdf5_experi_rey,
            swarm_feats=new_swarm_feats,
            num_boids=hdf5_experi_rey[reynolds_subfolders[0]].get("boids_pos").shape[1],
            range_steps=range_generate_and_optimize,
            behaviour_features_path="rey_swarm_metrics_30b.npz",
        )
        del cfg_experi_rey, hdf5_experi_rey, reynolds_subfolders

    if os.path.exists(load_npz_paths[7]):
        vic_swarm_metrics_30b = load_behaviour_features_from_npz(load_npz_paths[7])[
            "behaviour_swarm_metrics"
        ]
        for k, v in vic_swarm_metrics_30b.items():
            vic_swarm_metrics_30b[k] = v[
                :, range_generate_and_optimize.start : range_generate_and_optimize.stop
            ]
    else:
        cfg_experi_vic, hdf5_experi_vic, vicsek_subfolders = (
            processing_folder_structure(
                "/home/aj/Documents/repos/outputs/2025-06-06/02-59-01/", "vicsek"
            )
        )
        # TODO temporary since these config parts do not exist in earlier datalogs ...
        new_swarm_feats = None
        for _, temp_cfg in cfg_experi_vic.items():
            temp_cfg.metrics.swarm_mode_index = curr_cfg.metrics.swarm_mode_index
            temp_cfg.metrics.diffusion = curr_cfg.metrics.diffusion
            temp_cfg.metrics.neighbor_shortest_distances = (
                curr_cfg.metrics.neighbor_shortest_distances
            )
            temp_cfg.features = curr_cfg.features
            temp_cfg.features.LocalBoidsFeats.n_agents = (
                temp_cfg.behaviours.class_m.boid_count
            )
            temp_cfg.features.Kuckling2023.n_agents = (
                temp_cfg.behaviours.class_m.boid_count
            )
            if new_swarm_feats is None:
                new_swarm_feats = hydra.utils.instantiate(
                    temp_cfg.features
                ).get_selected()
        vic_swarm_metrics_30b = compute_behaviour_features(
            subfolder_names=vicsek_subfolders,
            experi_cfgs=cfg_experi_vic,
            experi_hdf5s=hdf5_experi_vic,
            swarm_feats=new_swarm_feats,
            num_boids=hdf5_experi_vic[vicsek_subfolders[0]].get("boids_pos").shape[1],
            range_steps=range_generate_and_optimize,
            behaviour_features_path="vic_swarm_metrics_30b.npz",
        )
        del cfg_experi_vic, hdf5_experi_vic, vicsek_subfolders

    if os.path.exists(load_npz_paths[8]):
        aggreg_swarm_metrics_30b = load_behaviour_features_from_npz(load_npz_paths[8])[
            "behaviour_swarm_metrics"
        ]
        for k, v in aggreg_swarm_metrics_30b.items():
            aggreg_swarm_metrics_30b[k] = v[
                :, range_generate_and_optimize.start : range_generate_and_optimize.stop
            ]
    else:
        cfg_experi_aggreg, hdf5_experi_aggreg, aggregation_subfolders = (
            processing_folder_structure(
                "/home/aj/Documents/repos/outputs/2025-06-06/07-57-24/",
                "aggregation",
            )
        )
        # TODO temporary since these config parts do not exist in earlier datalogs ...
        new_swarm_feats = None
        for _, temp_cfg in cfg_experi_aggreg.items():
            temp_cfg.metrics.swarm_mode_index = curr_cfg.metrics.swarm_mode_index
            temp_cfg.metrics.diffusion = curr_cfg.metrics.diffusion
            temp_cfg.metrics.neighbor_shortest_distances = (
                curr_cfg.metrics.neighbor_shortest_distances
            )
            temp_cfg.features = curr_cfg.features
            temp_cfg.features.LocalBoidsFeats.n_agents = (
                temp_cfg.behaviours.class_m.boid_count
            )
            temp_cfg.features.Kuckling2023.n_agents = (
                temp_cfg.behaviours.class_m.boid_count
            )
            if new_swarm_feats is None:
                new_swarm_feats = hydra.utils.instantiate(
                    temp_cfg.features
                ).get_selected()
        aggreg_swarm_metrics_30b = compute_behaviour_features(
            subfolder_names=aggregation_subfolders,
            experi_cfgs=cfg_experi_aggreg,
            experi_hdf5s=hdf5_experi_aggreg,
            swarm_feats=new_swarm_feats,
            num_boids=hdf5_experi_aggreg[aggregation_subfolders[0]]
            .get("boids_pos")
            .shape[1],
            range_steps=range_generate_and_optimize,
            behaviour_features_path="aggreg_swarm_metrics_30b.npz",
        )
        del cfg_experi_aggreg, hdf5_experi_aggreg, aggregation_subfolders

    if os.path.exists(load_npz_paths[9]):
        disper_swarm_metrics_30b = load_behaviour_features_from_npz(load_npz_paths[9])[
            "behaviour_swarm_metrics"
        ]
        for k, v in disper_swarm_metrics_30b.items():
            disper_swarm_metrics_30b[k] = v[
                :, range_generate_and_optimize.start : range_generate_and_optimize.stop
            ]
    else:
        cfg_experi_disper, hdf5_experi_disper, dispersion_subfolders = (
            processing_folder_structure(
                "/home/aj/Documents/repos/outputs/2025-06-06/11-47-03/", "dispersion"
            )
        )
        # TODO temporary since these config parts do not exist in earlier datalogs ...
        new_swarm_feats = None
        for _, temp_cfg in cfg_experi_disper.items():
            temp_cfg.metrics.swarm_mode_index = curr_cfg.metrics.swarm_mode_index
            temp_cfg.metrics.diffusion = curr_cfg.metrics.diffusion
            temp_cfg.metrics.neighbor_shortest_distances = (
                curr_cfg.metrics.neighbor_shortest_distances
            )
            temp_cfg.features = curr_cfg.features
            temp_cfg.features.LocalBoidsFeats.n_agents = (
                temp_cfg.behaviours.class_m.boid_count
            )
            temp_cfg.features.Kuckling2023.n_agents = (
                temp_cfg.behaviours.class_m.boid_count
            )
            if new_swarm_feats is None:
                new_swarm_feats = hydra.utils.instantiate(
                    temp_cfg.features
                ).get_selected()
        disper_swarm_metrics_30b = compute_behaviour_features(
            subfolder_names=dispersion_subfolders,
            experi_cfgs=cfg_experi_disper,
            experi_hdf5s=hdf5_experi_disper,
            swarm_feats=new_swarm_feats,
            num_boids=hdf5_experi_disper[dispersion_subfolders[0]]
            .get("boids_pos")
            .shape[1],
            range_steps=range_generate_and_optimize,
            behaviour_features_path="disper_swarm_metrics_30b.npz",
        )
        del cfg_experi_disper, hdf5_experi_disper, dispersion_subfolders

    if os.path.exists(load_npz_paths[10]):
        balli_swarm_metrics_30b = load_behaviour_features_from_npz(load_npz_paths[10])[
            "behaviour_swarm_metrics"
        ]
        for k, v in balli_swarm_metrics_30b.items():
            balli_swarm_metrics_30b[k] = v[
                :, range_generate_and_optimize.start : range_generate_and_optimize.stop
            ]
    else:
        cfg_experi_balli, hdf5_experi_balli, ballistic_subfolders = (
            processing_folder_structure(
                "/home/aj/Documents/repos/outputs/2025-06-12/04-27-18/",
                "ballistic",
            )
        )
        # TODO temporary since `max_dist` is used in a different context on `ballistic` and `brownian`
        new_swarm_feats = None
        for _, temp_cfg in cfg_experi_balli.items():
            temp_cfg.metrics.swarm_mode_index = curr_cfg.metrics.swarm_mode_index
            temp_cfg.metrics.diffusion = curr_cfg.metrics.diffusion
            temp_cfg.metrics.neighbor_shortest_distances = (
                curr_cfg.metrics.neighbor_shortest_distances
            )
            temp_cfg.features = curr_cfg.features
            temp_cfg.features.LocalBoidsFeats.n_agents = (
                temp_cfg.behaviours.class_m.boid_count
            )
            temp_cfg.features.Kuckling2023.n_agents = (
                temp_cfg.behaviours.class_m.boid_count
            )
            if new_swarm_feats is None:
                new_swarm_feats = hydra.utils.instantiate(
                    temp_cfg.features
                ).get_selected()
            temp_cfg.behaviours.class_m.model.max_dist = 100
        balli_swarm_metrics_30b = compute_behaviour_features(
            subfolder_names=ballistic_subfolders,
            experi_cfgs=cfg_experi_balli,
            experi_hdf5s=hdf5_experi_balli,
            swarm_feats=new_swarm_feats,
            num_boids=hdf5_experi_balli[ballistic_subfolders[0]]
            .get("boids_pos")
            .shape[1],
            range_steps=range_generate_and_optimize,
            behaviour_features_path="balli_swarm_metrics_30b.npz",
        )
        del cfg_experi_balli, hdf5_experi_balli, ballistic_subfolders

    if os.path.exists(load_npz_paths[11]):
        brown_swarm_metrics_30b = load_behaviour_features_from_npz(load_npz_paths[11])[
            "behaviour_swarm_metrics"
        ]
        for k, v in brown_swarm_metrics_30b.items():
            brown_swarm_metrics_30b[k] = v[
                :, range_generate_and_optimize.start : range_generate_and_optimize.stop
            ]
    else:
        cfg_experi_brown, hdf5_experi_brown, brownian_subfolders = (
            processing_folder_structure(
                "/home/aj/Documents/repos/outputs/2025-06-12/04-27-18/",
                "brownian",
            )
        )
        # TODO temporary since `max_dist` is used in a different context on `ballistic` and `brownian`
        new_swarm_feats = None
        for _, temp_cfg in cfg_experi_brown.items():
            temp_cfg.metrics.swarm_mode_index = curr_cfg.metrics.swarm_mode_index
            temp_cfg.metrics.diffusion = curr_cfg.metrics.diffusion
            temp_cfg.metrics.neighbor_shortest_distances = (
                curr_cfg.metrics.neighbor_shortest_distances
            )
            temp_cfg.features = curr_cfg.features
            temp_cfg.features.LocalBoidsFeats.n_agents = (
                temp_cfg.behaviours.class_m.boid_count
            )
            temp_cfg.features.Kuckling2023.n_agents = (
                temp_cfg.behaviours.class_m.boid_count
            )
            if new_swarm_feats is None:
                new_swarm_feats = hydra.utils.instantiate(
                    temp_cfg.features
                ).get_selected()
            temp_cfg.behaviours.class_m.model.max_dist = 100
        brown_swarm_metrics_30b = compute_behaviour_features(
            subfolder_names=brownian_subfolders,
            experi_cfgs=cfg_experi_brown,
            experi_hdf5s=hdf5_experi_brown,
            swarm_feats=new_swarm_feats,
            num_boids=hdf5_experi_brown[brownian_subfolders[0]]
            .get("boids_pos")
            .shape[1],
            range_steps=range_generate_and_optimize,
            behaviour_features_path="brown_swarm_metrics_30b.npz",
        )
        del cfg_experi_brown, hdf5_experi_brown, brownian_subfolders

    ######
    # Preparing `40u` data for Reynolds, Vicsek, Aggregation, Dispersion, Brownian and Ballistic Motion Random Walk
    ######
    if os.path.exists(load_npz_paths[12]):
        rey_swarm_metrics_40u = load_behaviour_features_from_npz(load_npz_paths[12])[
            "behaviour_swarm_metrics"
        ]
        for k, v in rey_swarm_metrics_40u.items():
            rey_swarm_metrics_40u[k] = v[
                :, range_generate_and_optimize.start : range_generate_and_optimize.stop
            ]
        num_rey_subfolders = list(rey_swarm_metrics_40u.values())[0].shape[0]
    else:
        cfg_experi_rey, hdf5_experi_rey, reynolds_subfolders = (
            processing_folder_structure(
                "/home/aj/Documents/repos/outputs/2025-06-06/00-11-53/", "reynolds"
            )
        )
        # TODO temporary since these config parts do not exist in earlier datalogs ...
        new_swarm_feats = None
        for _, temp_cfg in cfg_experi_rey.items():
            temp_cfg.metrics.swarm_mode_index = curr_cfg.metrics.swarm_mode_index
            temp_cfg.metrics.diffusion = curr_cfg.metrics.diffusion
            temp_cfg.metrics.neighbor_shortest_distances = (
                curr_cfg.metrics.neighbor_shortest_distances
            )
            temp_cfg.features = curr_cfg.features
            temp_cfg.features.LocalBoidsFeats.n_agents = (
                temp_cfg.behaviours.class_m.boid_count
            )
            temp_cfg.features.Kuckling2023.n_agents = (
                temp_cfg.behaviours.class_m.boid_count
            )
            if new_swarm_feats is None:
                new_swarm_feats = hydra.utils.instantiate(
                    temp_cfg.features
                ).get_selected()
        num_rey_subfolders = len(
            cfg_experi_rey[reynolds_subfolders[0]].seeds_per_exp_rep
        )
        rey_swarm_metrics_40u = compute_behaviour_features(
            subfolder_names=reynolds_subfolders,
            experi_cfgs=cfg_experi_rey,
            experi_hdf5s=hdf5_experi_rey,
            swarm_feats=new_swarm_feats,
            num_boids=hdf5_experi_rey[reynolds_subfolders[0]].get("boids_pos").shape[1],
            range_steps=range_generate_and_optimize,
            behaviour_features_path="rey_swarm_metrics_40u.npz",
        )
        del cfg_experi_rey, hdf5_experi_rey, reynolds_subfolders

    if os.path.exists(load_npz_paths[13]):
        vic_swarm_metrics_40u = load_behaviour_features_from_npz(load_npz_paths[13])[
            "behaviour_swarm_metrics"
        ]
        for k, v in vic_swarm_metrics_40u.items():
            vic_swarm_metrics_40u[k] = v[
                :, range_generate_and_optimize.start : range_generate_and_optimize.stop
            ]
    else:
        cfg_experi_vic, hdf5_experi_vic, vicsek_subfolders = (
            processing_folder_structure(
                "/home/aj/Documents/repos/outputs/2025-06-06/00-11-53/", "vicsek"
            )
        )
        # TODO temporary since these config parts do not exist in earlier datalogs ...
        new_swarm_feats = None
        for _, temp_cfg in cfg_experi_vic.items():
            temp_cfg.metrics.swarm_mode_index = curr_cfg.metrics.swarm_mode_index
            temp_cfg.metrics.diffusion = curr_cfg.metrics.diffusion
            temp_cfg.metrics.neighbor_shortest_distances = (
                curr_cfg.metrics.neighbor_shortest_distances
            )
            temp_cfg.features = curr_cfg.features
            temp_cfg.features.LocalBoidsFeats.n_agents = (
                temp_cfg.behaviours.class_m.boid_count
            )
            temp_cfg.features.Kuckling2023.n_agents = (
                temp_cfg.behaviours.class_m.boid_count
            )
            if new_swarm_feats is None:
                new_swarm_feats = hydra.utils.instantiate(
                    temp_cfg.features
                ).get_selected()
        vic_swarm_metrics_40u = compute_behaviour_features(
            subfolder_names=vicsek_subfolders,
            experi_cfgs=cfg_experi_vic,
            experi_hdf5s=hdf5_experi_vic,
            swarm_feats=new_swarm_feats,
            num_boids=hdf5_experi_vic[vicsek_subfolders[0]].get("boids_pos").shape[1],
            range_steps=range_generate_and_optimize,
            behaviour_features_path="vic_swarm_metrics_40u.npz",
        )
        del cfg_experi_vic, hdf5_experi_vic, vicsek_subfolders

    if os.path.exists(load_npz_paths[14]):
        aggreg_swarm_metrics_40u = load_behaviour_features_from_npz(load_npz_paths[14])[
            "behaviour_swarm_metrics"
        ]
        for k, v in aggreg_swarm_metrics_40u.items():
            aggreg_swarm_metrics_40u[k] = v[
                :, range_generate_and_optimize.start : range_generate_and_optimize.stop
            ]
    else:
        cfg_experi_aggreg, hdf5_experi_aggreg, aggregation_subfolders = (
            processing_folder_structure(
                "/home/aj/Documents/repos/outputs/2025-06-06/06-31-19/",
                "aggregation",
            )
        )
        # TODO temporary since these config parts do not exist in earlier datalogs ...
        new_swarm_feats = None
        for _, temp_cfg in cfg_experi_aggreg.items():
            temp_cfg.metrics.swarm_mode_index = curr_cfg.metrics.swarm_mode_index
            temp_cfg.metrics.diffusion = curr_cfg.metrics.diffusion
            temp_cfg.metrics.neighbor_shortest_distances = (
                curr_cfg.metrics.neighbor_shortest_distances
            )
            temp_cfg.features = curr_cfg.features
            temp_cfg.features.LocalBoidsFeats.n_agents = (
                temp_cfg.behaviours.class_m.boid_count
            )
            temp_cfg.features.Kuckling2023.n_agents = (
                temp_cfg.behaviours.class_m.boid_count
            )
            if new_swarm_feats is None:
                new_swarm_feats = hydra.utils.instantiate(
                    temp_cfg.features
                ).get_selected()
        aggreg_swarm_metrics_40u = compute_behaviour_features(
            subfolder_names=aggregation_subfolders,
            experi_cfgs=cfg_experi_aggreg,
            experi_hdf5s=hdf5_experi_aggreg,
            swarm_feats=new_swarm_feats,
            num_boids=hdf5_experi_aggreg[aggregation_subfolders[0]]
            .get("boids_pos")
            .shape[1],
            range_steps=range_generate_and_optimize,
            behaviour_features_path="aggreg_swarm_metrics_40u.npz",
        )
        del cfg_experi_aggreg, hdf5_experi_aggreg, aggregation_subfolders

    if os.path.exists(load_npz_paths[15]):
        disper_swarm_metrics_40u = load_behaviour_features_from_npz(load_npz_paths[15])[
            "behaviour_swarm_metrics"
        ]
        for k, v in disper_swarm_metrics_40u.items():
            disper_swarm_metrics_40u[k] = v[
                :, range_generate_and_optimize.start : range_generate_and_optimize.stop
            ]
    else:
        cfg_experi_disper, hdf5_experi_disper, dispersion_subfolders = (
            processing_folder_structure(
                "/home/aj/Documents/repos/outputs/2025-06-06/10-21-57/", "dispersion"
            )
        )
        # TODO temporary since these config parts do not exist in earlier datalogs ...
        new_swarm_feats = None
        for _, temp_cfg in cfg_experi_disper.items():
            temp_cfg.metrics.swarm_mode_index = curr_cfg.metrics.swarm_mode_index
            temp_cfg.metrics.diffusion = curr_cfg.metrics.diffusion
            temp_cfg.metrics.neighbor_shortest_distances = (
                curr_cfg.metrics.neighbor_shortest_distances
            )
            temp_cfg.features = curr_cfg.features
            temp_cfg.features.LocalBoidsFeats.n_agents = (
                temp_cfg.behaviours.class_m.boid_count
            )
            temp_cfg.features.Kuckling2023.n_agents = (
                temp_cfg.behaviours.class_m.boid_count
            )
            if new_swarm_feats is None:
                new_swarm_feats = hydra.utils.instantiate(
                    temp_cfg.features
                ).get_selected()
        disper_swarm_metrics_40u = compute_behaviour_features(
            subfolder_names=dispersion_subfolders,
            experi_cfgs=cfg_experi_disper,
            experi_hdf5s=hdf5_experi_disper,
            swarm_feats=new_swarm_feats,
            num_boids=hdf5_experi_disper[dispersion_subfolders[0]]
            .get("boids_pos")
            .shape[1],
            range_steps=range_generate_and_optimize,
            behaviour_features_path="disper_swarm_metrics_40u.npz",
        )
        del cfg_experi_disper, hdf5_experi_disper, dispersion_subfolders

    if os.path.exists(load_npz_paths[16]):
        balli_swarm_metrics_40u = load_behaviour_features_from_npz(load_npz_paths[16])[
            "behaviour_swarm_metrics"
        ]
        for k, v in balli_swarm_metrics_40u.items():
            balli_swarm_metrics_40u[k] = v[
                :, range_generate_and_optimize.start : range_generate_and_optimize.stop
            ]
    else:
        cfg_experi_balli, hdf5_experi_balli, ballistic_subfolders = (
            processing_folder_structure(
                "/home/aj/Documents/repos/outputs/2025-06-12/01-40-11/",
                "ballistic",
            )
        )
        # TODO temporary since `max_dist` is used in a different context on `ballistic` and `brownian`
        new_swarm_feats = None
        for _, temp_cfg in cfg_experi_balli.items():
            temp_cfg.metrics.swarm_mode_index = curr_cfg.metrics.swarm_mode_index
            temp_cfg.metrics.diffusion = curr_cfg.metrics.diffusion
            temp_cfg.metrics.neighbor_shortest_distances = (
                curr_cfg.metrics.neighbor_shortest_distances
            )
            temp_cfg.features = curr_cfg.features
            temp_cfg.features.LocalBoidsFeats.n_agents = (
                temp_cfg.behaviours.class_m.boid_count
            )
            temp_cfg.features.Kuckling2023.n_agents = (
                temp_cfg.behaviours.class_m.boid_count
            )
            if new_swarm_feats is None:
                new_swarm_feats = hydra.utils.instantiate(
                    temp_cfg.features
                ).get_selected()
            temp_cfg.behaviours.class_m.model.max_dist = 100
        balli_swarm_metrics_40u = compute_behaviour_features(
            subfolder_names=ballistic_subfolders,
            experi_cfgs=cfg_experi_balli,
            experi_hdf5s=hdf5_experi_balli,
            swarm_feats=new_swarm_feats,
            num_boids=hdf5_experi_balli[ballistic_subfolders[0]]
            .get("boids_pos")
            .shape[1],
            range_steps=range_generate_and_optimize,
            behaviour_features_path="balli_swarm_metrics_40u.npz",
        )
        del cfg_experi_balli, hdf5_experi_balli, ballistic_subfolders

    if os.path.exists(load_npz_paths[17]):
        brown_swarm_metrics_40u = load_behaviour_features_from_npz(load_npz_paths[17])[
            "behaviour_swarm_metrics"
        ]
        for k, v in brown_swarm_metrics_40u.items():
            brown_swarm_metrics_40u[k] = v[
                :, range_generate_and_optimize.start : range_generate_and_optimize.stop
            ]
    else:
        cfg_experi_brown, hdf5_experi_brown, brownian_subfolders = (
            processing_folder_structure(
                "/home/aj/Documents/repos/outputs/2025-06-12/01-40-11/",
                "brownian",
            )
        )
        # TODO temporary since `max_dist` is used in a different context on `ballistic` and `brownian`
        new_swarm_feats = None
        for _, temp_cfg in cfg_experi_brown.items():
            temp_cfg.metrics.swarm_mode_index = curr_cfg.metrics.swarm_mode_index
            temp_cfg.metrics.diffusion = curr_cfg.metrics.diffusion
            temp_cfg.metrics.neighbor_shortest_distances = (
                curr_cfg.metrics.neighbor_shortest_distances
            )
            temp_cfg.features = curr_cfg.features
            temp_cfg.features.LocalBoidsFeats.n_agents = (
                temp_cfg.behaviours.class_m.boid_count
            )
            temp_cfg.features.Kuckling2023.n_agents = (
                temp_cfg.behaviours.class_m.boid_count
            )
            if new_swarm_feats is None:
                new_swarm_feats = hydra.utils.instantiate(
                    temp_cfg.features
                ).get_selected()
            temp_cfg.behaviours.class_m.model.max_dist = 100
        brown_swarm_metrics_40u = compute_behaviour_features(
            subfolder_names=brownian_subfolders,
            experi_cfgs=cfg_experi_brown,
            experi_hdf5s=hdf5_experi_brown,
            swarm_feats=new_swarm_feats,
            num_boids=hdf5_experi_brown[brownian_subfolders[0]]
            .get("boids_pos")
            .shape[1],
            range_steps=range_generate_and_optimize,
            behaviour_features_path="brown_swarm_metrics_40u.npz",
        )
        del cfg_experi_brown, hdf5_experi_brown, brownian_subfolders

    ######
    # Normalize all behaviours simulations features to [0,1]
    ######
    for swarm_m_n in swarm_feats.names:
        rey_swarm_metrics_40b[swarm_m_n] = minmax_normalize_along_axis1(
            rey_swarm_metrics_40b[swarm_m_n]
        )
        rey_swarm_metrics_30b[swarm_m_n] = minmax_normalize_along_axis1(
            rey_swarm_metrics_30b[swarm_m_n]
        )
        rey_swarm_metrics_40u[swarm_m_n] = minmax_normalize_along_axis1(
            rey_swarm_metrics_40u[swarm_m_n]
        )
        vic_swarm_metrics_40b[swarm_m_n] = minmax_normalize_along_axis1(
            vic_swarm_metrics_40b[swarm_m_n]
        )
        vic_swarm_metrics_30b[swarm_m_n] = minmax_normalize_along_axis1(
            vic_swarm_metrics_30b[swarm_m_n]
        )
        vic_swarm_metrics_40u[swarm_m_n] = minmax_normalize_along_axis1(
            vic_swarm_metrics_40u[swarm_m_n]
        )
        aggreg_swarm_metrics_40b[swarm_m_n] = minmax_normalize_along_axis1(
            aggreg_swarm_metrics_40b[swarm_m_n]
        )
        aggreg_swarm_metrics_30b[swarm_m_n] = minmax_normalize_along_axis1(
            aggreg_swarm_metrics_30b[swarm_m_n]
        )
        aggreg_swarm_metrics_40u[swarm_m_n] = minmax_normalize_along_axis1(
            aggreg_swarm_metrics_40u[swarm_m_n]
        )
        disper_swarm_metrics_40b[swarm_m_n] = minmax_normalize_along_axis1(
            disper_swarm_metrics_40b[swarm_m_n]
        )
        disper_swarm_metrics_30b[swarm_m_n] = minmax_normalize_along_axis1(
            disper_swarm_metrics_30b[swarm_m_n]
        )
        disper_swarm_metrics_40u[swarm_m_n] = minmax_normalize_along_axis1(
            disper_swarm_metrics_40u[swarm_m_n]
        )
        balli_swarm_metrics_40b[swarm_m_n] = minmax_normalize_along_axis1(
            balli_swarm_metrics_40b[swarm_m_n]
        )
        balli_swarm_metrics_30b[swarm_m_n] = minmax_normalize_along_axis1(
            balli_swarm_metrics_30b[swarm_m_n]
        )
        balli_swarm_metrics_40u[swarm_m_n] = minmax_normalize_along_axis1(
            balli_swarm_metrics_40u[swarm_m_n]
        )
        brown_swarm_metrics_40b[swarm_m_n] = minmax_normalize_along_axis1(
            brown_swarm_metrics_40b[swarm_m_n]
        )
        brown_swarm_metrics_30b[swarm_m_n] = minmax_normalize_along_axis1(
            brown_swarm_metrics_30b[swarm_m_n]
        )
        brown_swarm_metrics_40u[swarm_m_n] = minmax_normalize_along_axis1(
            brown_swarm_metrics_40u[swarm_m_n]
        )

    return (
        rey_swarm_metrics_40b,
        rey_swarm_metrics_30b,
        rey_swarm_metrics_40u,
        vic_swarm_metrics_40b,
        vic_swarm_metrics_30b,
        vic_swarm_metrics_40u,
        aggreg_swarm_metrics_40b,
        aggreg_swarm_metrics_30b,
        aggreg_swarm_metrics_40u,
        disper_swarm_metrics_40b,
        disper_swarm_metrics_30b,
        disper_swarm_metrics_40u,
        balli_swarm_metrics_40b,
        balli_swarm_metrics_30b,
        balli_swarm_metrics_40u,
        brown_swarm_metrics_40b,
        brown_swarm_metrics_30b,
        brown_swarm_metrics_40u,
        rey_all_f_idxs,
    )


def get_cosine_sim_behaviours(
    reynolds_40b: np.ndarray,
    reynolds_30b: np.ndarray,
    reynolds_40u: np.ndarray,
    vicsek_40b: np.ndarray,
    vicsek_30b: np.ndarray,
    vicsek_40u: np.ndarray,
    aggregation_40b: np.ndarray,
    aggregation_30b: np.ndarray,
    aggregation_40u: np.ndarray,
    dispersion_40b: np.ndarray,
    dispersion_30b: np.ndarray,
    dispersion_40u: np.ndarray,
    ballistic_40b: np.ndarray,
    ballistic_30b: np.ndarray,
    ballistic_40u: np.ndarray,
    brownian_40b: np.ndarray,
    brownian_30b: np.ndarray,
    brownian_40u: np.ndarray,
    similarity_metrics: tuple,
    csv_path_name: str,
):
    """"""
    log.info("\n" + 20 * "=" + "\n")

    reynolds_to_reynolds_similarity = np.array(
        [
            compute_cosine_similarity_measure(
                reynolds_40b,
                reynolds_40b,
            ),
            compute_cosine_similarity_measure(
                reynolds_30b,
                reynolds_30b,
            ),
            compute_cosine_similarity_measure(
                reynolds_40u,
                reynolds_40u,
            ),
        ]
    )
    reynolds_to_vicsek_similarity = np.array(
        [
            compute_cosine_similarity_measure(
                reynolds_40b,
                vicsek_40b,
            ),
            compute_cosine_similarity_measure(
                reynolds_30b,
                vicsek_30b,
            ),
            compute_cosine_similarity_measure(
                reynolds_40u,
                vicsek_40u,
            ),
        ]
    )
    reynolds_to_aggregation_similarity = np.array(
        [
            compute_cosine_similarity_measure(
                reynolds_40b,
                aggregation_40b,
            ),
            compute_cosine_similarity_measure(
                reynolds_30b,
                aggregation_30b,
            ),
            compute_cosine_similarity_measure(
                reynolds_40u,
                aggregation_40u,
            ),
        ]
    )
    reynolds_to_dispersion_similarity = np.array(
        [
            compute_cosine_similarity_measure(
                reynolds_40b,
                dispersion_40b,
            ),
            compute_cosine_similarity_measure(
                reynolds_30b,
                dispersion_30b,
            ),
            compute_cosine_similarity_measure(
                reynolds_40u,
                dispersion_40u,
            ),
        ]
    )
    reynolds_to_ballistic_similarity = np.array(
        [
            compute_cosine_similarity_measure(
                reynolds_40b,
                ballistic_40b,
            ),
            compute_cosine_similarity_measure(
                reynolds_30b,
                ballistic_30b,
            ),
            compute_cosine_similarity_measure(
                reynolds_40u,
                ballistic_40u,
            ),
        ]
    )
    reynolds_to_brownian_similarity = np.array(
        [
            compute_cosine_similarity_measure(
                reynolds_40b,
                brownian_40b,
            ),
            compute_cosine_similarity_measure(
                reynolds_30b,
                brownian_30b,
            ),
            compute_cosine_similarity_measure(
                reynolds_40u,
                brownian_40u,
            ),
        ]
    )
    vicsek_to_vicsek_similarity = np.array(
        [
            compute_cosine_similarity_measure(
                vicsek_40b,
                vicsek_40b,
            ),
            compute_cosine_similarity_measure(
                vicsek_30b,
                vicsek_30b,
            ),
            compute_cosine_similarity_measure(
                vicsek_40u,
                vicsek_40u,
            ),
        ]
    )
    vicsek_to_aggregation_similarity = np.array(
        [
            compute_cosine_similarity_measure(
                vicsek_40b,
                aggregation_40b,
            ),
            compute_cosine_similarity_measure(
                vicsek_30b,
                aggregation_30b,
            ),
            compute_cosine_similarity_measure(
                vicsek_40u,
                aggregation_40u,
            ),
        ]
    )
    vicsek_to_dispersion_similarity = np.array(
        [
            compute_cosine_similarity_measure(
                vicsek_40b,
                dispersion_40b,
            ),
            compute_cosine_similarity_measure(
                vicsek_30b,
                dispersion_30b,
            ),
            compute_cosine_similarity_measure(
                vicsek_40u,
                dispersion_40u,
            ),
        ]
    )
    vicsek_to_ballistic_similarity = np.array(
        [
            compute_cosine_similarity_measure(
                vicsek_40b,
                ballistic_40b,
            ),
            compute_cosine_similarity_measure(
                vicsek_30b,
                ballistic_30b,
            ),
            compute_cosine_similarity_measure(
                vicsek_40u,
                ballistic_40u,
            ),
        ]
    )
    vicsek_to_brownian_similarity = np.array(
        [
            compute_cosine_similarity_measure(
                vicsek_40b,
                brownian_40b,
            ),
            compute_cosine_similarity_measure(
                vicsek_30b,
                brownian_30b,
            ),
            compute_cosine_similarity_measure(
                vicsek_40u,
                brownian_40u,
            ),
        ]
    )
    aggregation_to_aggregation_similarity = np.array(
        [
            compute_cosine_similarity_measure(
                aggregation_40b,
                aggregation_40b,
            ),
            compute_cosine_similarity_measure(
                aggregation_30b,
                aggregation_30b,
            ),
            compute_cosine_similarity_measure(
                aggregation_40u,
                aggregation_40u,
            ),
        ]
    )
    aggregation_to_dispersion_similarity = np.array(
        [
            compute_cosine_similarity_measure(
                aggregation_40b,
                dispersion_40b,
            ),
            compute_cosine_similarity_measure(
                aggregation_30b,
                dispersion_30b,
            ),
            compute_cosine_similarity_measure(
                aggregation_40u,
                dispersion_40u,
            ),
        ]
    )
    aggregation_to_ballistic_similarity = np.array(
        [
            compute_cosine_similarity_measure(
                aggregation_40b,
                ballistic_40b,
            ),
            compute_cosine_similarity_measure(
                aggregation_30b,
                ballistic_30b,
            ),
            compute_cosine_similarity_measure(
                aggregation_40u,
                ballistic_40u,
            ),
        ]
    )
    aggregation_to_brownian_similarity = np.array(
        [
            compute_cosine_similarity_measure(
                aggregation_40b,
                brownian_40b,
            ),
            compute_cosine_similarity_measure(
                aggregation_30b,
                brownian_30b,
            ),
            compute_cosine_similarity_measure(
                aggregation_40u,
                brownian_40u,
            ),
        ]
    )
    dispersion_to_dispersion_similarity = np.array(
        [
            compute_cosine_similarity_measure(
                dispersion_40b,
                dispersion_40b,
            ),
            compute_cosine_similarity_measure(
                dispersion_30b,
                dispersion_30b,
            ),
            compute_cosine_similarity_measure(
                dispersion_40u,
                dispersion_40u,
            ),
        ]
    )
    dispersion_to_ballistic_similarity = np.array(
        [
            compute_cosine_similarity_measure(
                dispersion_40b,
                ballistic_40b,
            ),
            compute_cosine_similarity_measure(
                dispersion_30b,
                ballistic_30b,
            ),
            compute_cosine_similarity_measure(
                dispersion_40u,
                ballistic_40u,
            ),
        ]
    )
    dispersion_to_brownian_similarity = np.array(
        [
            compute_cosine_similarity_measure(
                dispersion_40b,
                brownian_40b,
            ),
            compute_cosine_similarity_measure(
                dispersion_30b,
                brownian_30b,
            ),
            compute_cosine_similarity_measure(
                dispersion_40u,
                brownian_40u,
            ),
        ]
    )
    ballistic_to_ballistic_similarity = np.array(
        [
            compute_cosine_similarity_measure(
                ballistic_40b,
                ballistic_40b,
            ),
            compute_cosine_similarity_measure(
                ballistic_30b,
                ballistic_30b,
            ),
            compute_cosine_similarity_measure(
                ballistic_40u,
                ballistic_40u,
            ),
        ]
    )
    ballistic_to_brownian_similarity = np.array(
        [
            compute_cosine_similarity_measure(
                ballistic_40b,
                brownian_40b,
            ),
            compute_cosine_similarity_measure(
                ballistic_30b,
                brownian_30b,
            ),
            compute_cosine_similarity_measure(
                ballistic_40u,
                brownian_40u,
            ),
        ]
    )
    brownian_to_brownian_similarity = np.array(
        [
            compute_cosine_similarity_measure(
                brownian_40b,
                brownian_40b,
            ),
            compute_cosine_similarity_measure(
                brownian_30b,
                brownian_30b,
            ),
            compute_cosine_similarity_measure(
                brownian_40u,
                brownian_40u,
            ),
        ]
    )

    df = open_dataframe(
        csv_path_name,
        [
            "similarity_type",
            "metrics",
            "similarity_score_mean",
            "similarity_score_std",
        ],
    )
    df = append_row_to_dataframe(
        df,
        {
            "similarity_type": "Reynolds to Reynolds",
            "metrics": similarity_metrics,
            "similarity_score_mean": np.round(
                np.mean(reynolds_to_reynolds_similarity), 2
            ),
            "similarity_score_std": np.round(
                np.std(reynolds_to_reynolds_similarity), 2
            ),
        },
    )
    df = append_row_to_dataframe(
        df,
        {
            "similarity_type": "Reynolds to Vicsek",
            "metrics": similarity_metrics,
            "similarity_score_mean": np.round(
                np.mean(reynolds_to_vicsek_similarity), 2
            ),
            "similarity_score_std": np.round(np.std(reynolds_to_vicsek_similarity), 2),
        },
    )
    df = append_row_to_dataframe(
        df,
        {
            "similarity_type": "Reynolds to Aggregation",
            "metrics": similarity_metrics,
            "similarity_score_mean": np.round(
                np.mean(reynolds_to_aggregation_similarity), 2
            ),
            "similarity_score_std": np.round(
                np.std(reynolds_to_aggregation_similarity), 2
            ),
        },
    )
    df = append_row_to_dataframe(
        df,
        {
            "similarity_type": "Reynolds to Dispersion",
            "metrics": similarity_metrics,
            "similarity_score_mean": np.round(
                np.mean(reynolds_to_dispersion_similarity), 2
            ),
            "similarity_score_std": np.round(
                np.std(reynolds_to_dispersion_similarity), 2
            ),
        },
    )
    df = append_row_to_dataframe(
        df,
        {
            "similarity_type": "Reynolds to Ballistic",
            "metrics": similarity_metrics,
            "similarity_score_mean": np.round(
                np.mean(reynolds_to_ballistic_similarity), 2
            ),
            "similarity_score_std": np.round(
                np.std(reynolds_to_ballistic_similarity), 2
            ),
        },
    )
    df = append_row_to_dataframe(
        df,
        {
            "similarity_type": "Reynolds to Brownian",
            "metrics": similarity_metrics,
            "similarity_score_mean": np.round(
                np.mean(reynolds_to_brownian_similarity), 2
            ),
            "similarity_score_std": np.round(
                np.std(reynolds_to_brownian_similarity), 2
            ),
        },
    )
    df = append_row_to_dataframe(
        df,
        {
            "similarity_type": "Vicsek to Vicsek",
            "metrics": similarity_metrics,
            "similarity_score_mean": np.round(np.mean(vicsek_to_vicsek_similarity), 2),
            "similarity_score_std": np.round(np.std(vicsek_to_vicsek_similarity), 2),
        },
    )
    df = append_row_to_dataframe(
        df,
        {
            "similarity_type": "Vicsek to Aggregation",
            "metrics": similarity_metrics,
            "similarity_score_mean": np.round(
                np.mean(vicsek_to_aggregation_similarity), 2
            ),
            "similarity_score_std": np.round(
                np.std(vicsek_to_aggregation_similarity), 2
            ),
        },
    )
    df = append_row_to_dataframe(
        df,
        {
            "similarity_type": "Vicsek to Dispersion",
            "metrics": similarity_metrics,
            "similarity_score_mean": np.round(
                np.mean(vicsek_to_dispersion_similarity), 2
            ),
            "similarity_score_std": np.round(
                np.std(vicsek_to_dispersion_similarity), 2
            ),
        },
    )
    df = append_row_to_dataframe(
        df,
        {
            "similarity_type": "Vicsek to Ballistic",
            "metrics": similarity_metrics,
            "similarity_score_mean": np.round(
                np.mean(vicsek_to_ballistic_similarity), 2
            ),
            "similarity_score_std": np.round(np.std(vicsek_to_ballistic_similarity), 2),
        },
    )
    df = append_row_to_dataframe(
        df,
        {
            "similarity_type": "Vicsek to Brownian",
            "metrics": similarity_metrics,
            "similarity_score_mean": np.round(
                np.mean(vicsek_to_brownian_similarity), 2
            ),
            "similarity_score_std": np.round(np.std(vicsek_to_brownian_similarity), 2),
        },
    )
    df = append_row_to_dataframe(
        df,
        {
            "similarity_type": "Aggregation to Aggregation",
            "metrics": similarity_metrics,
            "similarity_score_mean": np.round(
                np.mean(aggregation_to_aggregation_similarity), 2
            ),
            "similarity_score_std": np.round(
                np.std(aggregation_to_aggregation_similarity), 2
            ),
        },
    )
    df = append_row_to_dataframe(
        df,
        {
            "similarity_type": "Aggregation to Dispersion",
            "metrics": similarity_metrics,
            "similarity_score_mean": np.round(
                np.mean(aggregation_to_dispersion_similarity), 2
            ),
            "similarity_score_std": np.round(
                np.std(aggregation_to_dispersion_similarity), 2
            ),
        },
    )
    df = append_row_to_dataframe(
        df,
        {
            "similarity_type": "Aggregation to Ballistic",
            "metrics": similarity_metrics,
            "similarity_score_mean": np.round(
                np.mean(aggregation_to_ballistic_similarity), 2
            ),
            "similarity_score_std": np.round(
                np.std(aggregation_to_ballistic_similarity), 2
            ),
        },
    )
    df = append_row_to_dataframe(
        df,
        {
            "similarity_type": "Aggregation to Brownian",
            "metrics": similarity_metrics,
            "similarity_score_mean": np.round(
                np.mean(aggregation_to_brownian_similarity), 2
            ),
            "similarity_score_std": np.round(
                np.std(aggregation_to_brownian_similarity), 2
            ),
        },
    )
    df = append_row_to_dataframe(
        df,
        {
            "similarity_type": "Dispersion to Dispersion",
            "metrics": similarity_metrics,
            "similarity_score_mean": np.round(
                np.mean(dispersion_to_dispersion_similarity), 2
            ),
            "similarity_score_std": np.round(
                np.std(dispersion_to_dispersion_similarity), 2
            ),
        },
    )
    df = append_row_to_dataframe(
        df,
        {
            "similarity_type": "Dispersion to Ballistic",
            "metrics": similarity_metrics,
            "similarity_score_mean": np.round(
                np.mean(dispersion_to_ballistic_similarity), 2
            ),
            "similarity_score_std": np.round(
                np.std(dispersion_to_ballistic_similarity), 2
            ),
        },
    )
    df = append_row_to_dataframe(
        df,
        {
            "similarity_type": "Dispersion to Brownian",
            "metrics": similarity_metrics,
            "similarity_score_mean": np.round(
                np.mean(dispersion_to_brownian_similarity), 2
            ),
            "similarity_score_std": np.round(
                np.std(dispersion_to_brownian_similarity), 2
            ),
        },
    )
    df = append_row_to_dataframe(
        df,
        {
            "similarity_type": "Ballistic to Ballistic",
            "metrics": similarity_metrics,
            "similarity_score_mean": np.round(
                np.mean(ballistic_to_ballistic_similarity), 2
            ),
            "similarity_score_std": np.round(
                np.std(ballistic_to_ballistic_similarity), 2
            ),
        },
    )
    df = append_row_to_dataframe(
        df,
        {
            "similarity_type": "Ballistic to Brownian",
            "metrics": similarity_metrics,
            "similarity_score_mean": np.round(
                np.mean(ballistic_to_brownian_similarity), 2
            ),
            "similarity_score_std": np.round(
                np.std(ballistic_to_brownian_similarity), 2
            ),
        },
    )
    df = append_row_to_dataframe(
        df,
        {
            "similarity_type": "Brownian to Brownian",
            "metrics": similarity_metrics,
            "similarity_score_mean": np.round(
                np.mean(brownian_to_brownian_similarity), 2
            ),
            "similarity_score_std": np.round(
                np.std(brownian_to_brownian_similarity), 2
            ),
        },
    )
    export_dataframe(df, csv_path_name, latex_columns=None)

    log.info("\n" + 20 * "=" + "\n")


def get_euclidean_distance_sim_behaviours(
    reynolds_40b: np.ndarray,
    reynolds_30b: np.ndarray,
    reynolds_40u: np.ndarray,
    vicsek_40b: np.ndarray,
    vicsek_30b: np.ndarray,
    vicsek_40u: np.ndarray,
    aggregation_40b: np.ndarray,
    aggregation_30b: np.ndarray,
    aggregation_40u: np.ndarray,
    dispersion_40b: np.ndarray,
    dispersion_30b: np.ndarray,
    dispersion_40u: np.ndarray,
    ballistic_40b: np.ndarray,
    ballistic_30b: np.ndarray,
    ballistic_40u: np.ndarray,
    brownian_40b: np.ndarray,
    brownian_30b: np.ndarray,
    brownian_40u: np.ndarray,
    similarity_metrics: tuple,
    csv_path_name: str,
    cfg: DictConfig,
):
    """"""
    log.info("\n" + 20 * "=" + "\n")

    reynolds_to_reynolds_similarity = np.array(
        [
            compute_euclidean_distance_measure(
                reynolds_40b,
                reynolds_40b,
                cfg.behaviours.class_m.walls_b,
                cfg.behaviours.class_m.box_size,
            ),
            compute_euclidean_distance_measure(
                reynolds_30b,
                reynolds_30b,
                cfg.behaviours.class_m.walls_b,
                cfg.behaviours.class_m.box_size,
            ),
            compute_euclidean_distance_measure(
                reynolds_40u,
                reynolds_40u,
                cfg.behaviours.class_m.walls_b,
                cfg.behaviours.class_m.box_size,
            ),
        ]
    )
    reynolds_to_vicsek_similarity = np.array(
        [
            compute_euclidean_distance_measure(
                reynolds_40b,
                vicsek_40b,
                cfg.behaviours.class_m.walls_b,
                cfg.behaviours.class_m.box_size,
            ),
            compute_euclidean_distance_measure(
                reynolds_30b,
                vicsek_30b,
                cfg.behaviours.class_m.walls_b,
                cfg.behaviours.class_m.box_size,
            ),
            compute_euclidean_distance_measure(
                reynolds_40u,
                vicsek_40u,
                cfg.behaviours.class_m.walls_b,
                cfg.behaviours.class_m.box_size,
            ),
        ]
    )
    reynolds_to_aggregation_similarity = np.array(
        [
            compute_euclidean_distance_measure(
                reynolds_40b,
                aggregation_40b,
                cfg.behaviours.class_m.walls_b,
                cfg.behaviours.class_m.box_size,
            ),
            compute_euclidean_distance_measure(
                reynolds_30b,
                aggregation_30b,
                cfg.behaviours.class_m.walls_b,
                cfg.behaviours.class_m.box_size,
            ),
            compute_euclidean_distance_measure(
                reynolds_40u,
                aggregation_40u,
                cfg.behaviours.class_m.walls_b,
                cfg.behaviours.class_m.box_size,
            ),
        ]
    )
    reynolds_to_dispersion_similarity = np.array(
        [
            compute_euclidean_distance_measure(
                reynolds_40b,
                dispersion_40b,
                cfg.behaviours.class_m.walls_b,
                cfg.behaviours.class_m.box_size,
            ),
            compute_euclidean_distance_measure(
                reynolds_30b,
                dispersion_30b,
                cfg.behaviours.class_m.walls_b,
                cfg.behaviours.class_m.box_size,
            ),
            compute_euclidean_distance_measure(
                reynolds_40u,
                dispersion_40u,
                cfg.behaviours.class_m.walls_b,
                cfg.behaviours.class_m.box_size,
            ),
        ]
    )
    reynolds_to_ballistic_similarity = np.array(
        [
            compute_euclidean_distance_measure(
                reynolds_40b,
                ballistic_40b,
                cfg.behaviours.class_m.walls_b,
                cfg.behaviours.class_m.box_size,
            ),
            compute_euclidean_distance_measure(
                reynolds_30b,
                ballistic_30b,
                cfg.behaviours.class_m.walls_b,
                cfg.behaviours.class_m.box_size,
            ),
            compute_euclidean_distance_measure(
                reynolds_40u,
                ballistic_40u,
                cfg.behaviours.class_m.walls_b,
                cfg.behaviours.class_m.box_size,
            ),
        ]
    )
    reynolds_to_brownian_similarity = np.array(
        [
            compute_euclidean_distance_measure(
                reynolds_40b,
                brownian_40b,
                cfg.behaviours.class_m.walls_b,
                cfg.behaviours.class_m.box_size,
            ),
            compute_euclidean_distance_measure(
                reynolds_30b,
                brownian_30b,
                cfg.behaviours.class_m.walls_b,
                cfg.behaviours.class_m.box_size,
            ),
            compute_euclidean_distance_measure(
                reynolds_40u,
                brownian_40u,
                cfg.behaviours.class_m.walls_b,
                cfg.behaviours.class_m.box_size,
            ),
        ]
    )
    vicsek_to_vicsek_similarity = np.array(
        [
            compute_euclidean_distance_measure(
                vicsek_40b,
                vicsek_40b,
                cfg.behaviours.class_m.walls_b,
                cfg.behaviours.class_m.box_size,
            ),
            compute_euclidean_distance_measure(
                vicsek_30b,
                vicsek_30b,
                cfg.behaviours.class_m.walls_b,
                cfg.behaviours.class_m.box_size,
            ),
            compute_euclidean_distance_measure(
                vicsek_40u,
                vicsek_40u,
                cfg.behaviours.class_m.walls_b,
                cfg.behaviours.class_m.box_size,
            ),
        ]
    )
    vicsek_to_aggregation_similarity = np.array(
        [
            compute_euclidean_distance_measure(
                vicsek_40b,
                aggregation_40b,
                cfg.behaviours.class_m.walls_b,
                cfg.behaviours.class_m.box_size,
            ),
            compute_euclidean_distance_measure(
                vicsek_30b,
                aggregation_30b,
                cfg.behaviours.class_m.walls_b,
                cfg.behaviours.class_m.box_size,
            ),
            compute_euclidean_distance_measure(
                vicsek_40u,
                aggregation_40u,
                cfg.behaviours.class_m.walls_b,
                cfg.behaviours.class_m.box_size,
            ),
        ]
    )
    vicsek_to_dispersion_similarity = np.array(
        [
            compute_euclidean_distance_measure(
                vicsek_40b,
                dispersion_40b,
                cfg.behaviours.class_m.walls_b,
                cfg.behaviours.class_m.box_size,
            ),
            compute_euclidean_distance_measure(
                vicsek_30b,
                dispersion_30b,
                cfg.behaviours.class_m.walls_b,
                cfg.behaviours.class_m.box_size,
            ),
            compute_euclidean_distance_measure(
                vicsek_40u,
                dispersion_40u,
                cfg.behaviours.class_m.walls_b,
                cfg.behaviours.class_m.box_size,
            ),
        ]
    )
    vicsek_to_ballistic_similarity = np.array(
        [
            compute_euclidean_distance_measure(
                vicsek_40b,
                ballistic_40b,
                cfg.behaviours.class_m.walls_b,
                cfg.behaviours.class_m.box_size,
            ),
            compute_euclidean_distance_measure(
                vicsek_30b,
                ballistic_30b,
                cfg.behaviours.class_m.walls_b,
                cfg.behaviours.class_m.box_size,
            ),
            compute_euclidean_distance_measure(
                vicsek_40u,
                ballistic_40u,
                cfg.behaviours.class_m.walls_b,
                cfg.behaviours.class_m.box_size,
            ),
        ]
    )
    vicsek_to_brownian_similarity = np.array(
        [
            compute_euclidean_distance_measure(
                vicsek_40b,
                brownian_40b,
                cfg.behaviours.class_m.walls_b,
                cfg.behaviours.class_m.box_size,
            ),
            compute_euclidean_distance_measure(
                vicsek_30b,
                brownian_30b,
                cfg.behaviours.class_m.walls_b,
                cfg.behaviours.class_m.box_size,
            ),
            compute_euclidean_distance_measure(
                vicsek_40u,
                brownian_40u,
                cfg.behaviours.class_m.walls_b,
                cfg.behaviours.class_m.box_size,
            ),
        ]
    )
    aggregation_to_aggregation_similarity = np.array(
        [
            compute_euclidean_distance_measure(
                aggregation_40b,
                aggregation_40b,
                cfg.behaviours.class_m.walls_b,
                cfg.behaviours.class_m.box_size,
            ),
            compute_euclidean_distance_measure(
                aggregation_30b,
                aggregation_30b,
                cfg.behaviours.class_m.walls_b,
                cfg.behaviours.class_m.box_size,
            ),
            compute_euclidean_distance_measure(
                aggregation_40u,
                aggregation_40u,
                cfg.behaviours.class_m.walls_b,
                cfg.behaviours.class_m.box_size,
            ),
        ]
    )
    aggregation_to_dispersion_similarity = np.array(
        [
            compute_euclidean_distance_measure(
                aggregation_40b,
                dispersion_40b,
                cfg.behaviours.class_m.walls_b,
                cfg.behaviours.class_m.box_size,
            ),
            compute_euclidean_distance_measure(
                aggregation_30b,
                dispersion_30b,
                cfg.behaviours.class_m.walls_b,
                cfg.behaviours.class_m.box_size,
            ),
            compute_euclidean_distance_measure(
                aggregation_40u,
                dispersion_40u,
                cfg.behaviours.class_m.walls_b,
                cfg.behaviours.class_m.box_size,
            ),
        ]
    )
    aggregation_to_ballistic_similarity = np.array(
        [
            compute_euclidean_distance_measure(
                aggregation_40b,
                ballistic_40b,
                cfg.behaviours.class_m.walls_b,
                cfg.behaviours.class_m.box_size,
            ),
            compute_euclidean_distance_measure(
                aggregation_30b,
                ballistic_30b,
                cfg.behaviours.class_m.walls_b,
                cfg.behaviours.class_m.box_size,
            ),
            compute_euclidean_distance_measure(
                aggregation_40u,
                ballistic_40u,
                cfg.behaviours.class_m.walls_b,
                cfg.behaviours.class_m.box_size,
            ),
        ]
    )
    aggregation_to_brownian_similarity = np.array(
        [
            compute_euclidean_distance_measure(
                aggregation_40b,
                brownian_40b,
                cfg.behaviours.class_m.walls_b,
                cfg.behaviours.class_m.box_size,
            ),
            compute_euclidean_distance_measure(
                aggregation_30b,
                brownian_30b,
                cfg.behaviours.class_m.walls_b,
                cfg.behaviours.class_m.box_size,
            ),
            compute_euclidean_distance_measure(
                aggregation_40u,
                brownian_40u,
                cfg.behaviours.class_m.walls_b,
                cfg.behaviours.class_m.box_size,
            ),
        ]
    )
    dispersion_to_dispersion_similarity = np.array(
        [
            compute_euclidean_distance_measure(
                dispersion_40b,
                dispersion_40b,
                cfg.behaviours.class_m.walls_b,
                cfg.behaviours.class_m.box_size,
            ),
            compute_euclidean_distance_measure(
                dispersion_30b,
                dispersion_30b,
                cfg.behaviours.class_m.walls_b,
                cfg.behaviours.class_m.box_size,
            ),
            compute_euclidean_distance_measure(
                dispersion_40u,
                dispersion_40u,
                cfg.behaviours.class_m.walls_b,
                cfg.behaviours.class_m.box_size,
            ),
        ]
    )
    dispersion_to_ballistic_similarity = np.array(
        [
            compute_euclidean_distance_measure(
                dispersion_40b,
                ballistic_40b,
                cfg.behaviours.class_m.walls_b,
                cfg.behaviours.class_m.box_size,
            ),
            compute_euclidean_distance_measure(
                dispersion_30b,
                ballistic_30b,
                cfg.behaviours.class_m.walls_b,
                cfg.behaviours.class_m.box_size,
            ),
            compute_euclidean_distance_measure(
                dispersion_40u,
                ballistic_40u,
                cfg.behaviours.class_m.walls_b,
                cfg.behaviours.class_m.box_size,
            ),
        ]
    )
    dispersion_to_brownian_similarity = np.array(
        [
            compute_euclidean_distance_measure(
                dispersion_40b,
                brownian_40b,
                cfg.behaviours.class_m.walls_b,
                cfg.behaviours.class_m.box_size,
            ),
            compute_euclidean_distance_measure(
                dispersion_30b,
                brownian_30b,
                cfg.behaviours.class_m.walls_b,
                cfg.behaviours.class_m.box_size,
            ),
            compute_euclidean_distance_measure(
                dispersion_40u,
                brownian_40u,
                cfg.behaviours.class_m.walls_b,
                cfg.behaviours.class_m.box_size,
            ),
        ]
    )
    ballistic_to_ballistic_similarity = np.array(
        [
            compute_euclidean_distance_measure(
                ballistic_40b,
                ballistic_40b,
                cfg.behaviours.class_m.walls_b,
                cfg.behaviours.class_m.box_size,
            ),
            compute_euclidean_distance_measure(
                ballistic_30b,
                ballistic_30b,
                cfg.behaviours.class_m.walls_b,
                cfg.behaviours.class_m.box_size,
            ),
            compute_euclidean_distance_measure(
                ballistic_40u,
                ballistic_40u,
                cfg.behaviours.class_m.walls_b,
                cfg.behaviours.class_m.box_size,
            ),
        ]
    )
    ballistic_to_brownian_similarity = np.array(
        [
            compute_euclidean_distance_measure(
                ballistic_40b,
                brownian_40b,
                cfg.behaviours.class_m.walls_b,
                cfg.behaviours.class_m.box_size,
            ),
            compute_euclidean_distance_measure(
                ballistic_30b,
                brownian_30b,
                cfg.behaviours.class_m.walls_b,
                cfg.behaviours.class_m.box_size,
            ),
            compute_euclidean_distance_measure(
                ballistic_40u,
                brownian_40u,
                cfg.behaviours.class_m.walls_b,
                cfg.behaviours.class_m.box_size,
            ),
        ]
    )
    brownian_to_brownian_similarity = np.array(
        [
            compute_euclidean_distance_measure(
                brownian_40b,
                brownian_40b,
                cfg.behaviours.class_m.walls_b,
                cfg.behaviours.class_m.box_size,
            ),
            compute_euclidean_distance_measure(
                brownian_30b,
                brownian_30b,
                cfg.behaviours.class_m.walls_b,
                cfg.behaviours.class_m.box_size,
            ),
            compute_euclidean_distance_measure(
                brownian_40u,
                brownian_40u,
                cfg.behaviours.class_m.walls_b,
                cfg.behaviours.class_m.box_size,
            ),
        ]
    )

    df = open_dataframe(
        csv_path_name,
        [
            "similarity_type",
            "metrics",
            "similarity_score_mean",
            "similarity_score_std",
        ],
    )
    df = append_row_to_dataframe(
        df,
        {
            "similarity_type": "Reynolds to Reynolds",
            "metrics": similarity_metrics,
            "similarity_score_mean": np.round(
                np.mean(reynolds_to_reynolds_similarity), 2
            ),
            "similarity_score_std": np.round(
                np.std(reynolds_to_reynolds_similarity), 2
            ),
        },
    )
    df = append_row_to_dataframe(
        df,
        {
            "similarity_type": "Reynolds to Vicsek",
            "metrics": similarity_metrics,
            "similarity_score_mean": np.round(
                np.mean(reynolds_to_vicsek_similarity), 2
            ),
            "similarity_score_std": np.round(np.std(reynolds_to_vicsek_similarity), 2),
        },
    )
    df = append_row_to_dataframe(
        df,
        {
            "similarity_type": "Reynolds to Aggregation",
            "metrics": similarity_metrics,
            "similarity_score_mean": np.round(
                np.mean(reynolds_to_aggregation_similarity), 2
            ),
            "similarity_score_std": np.round(
                np.std(reynolds_to_aggregation_similarity), 2
            ),
        },
    )
    df = append_row_to_dataframe(
        df,
        {
            "similarity_type": "Reynolds to Dispersion",
            "metrics": similarity_metrics,
            "similarity_score_mean": np.round(
                np.mean(reynolds_to_dispersion_similarity), 2
            ),
            "similarity_score_std": np.round(
                np.std(reynolds_to_dispersion_similarity), 2
            ),
        },
    )
    df = append_row_to_dataframe(
        df,
        {
            "similarity_type": "Reynolds to Ballistic",
            "metrics": similarity_metrics,
            "similarity_score_mean": np.round(
                np.mean(reynolds_to_ballistic_similarity), 2
            ),
            "similarity_score_std": np.round(
                np.std(reynolds_to_ballistic_similarity), 2
            ),
        },
    )
    df = append_row_to_dataframe(
        df,
        {
            "similarity_type": "Reynolds to Brownian",
            "metrics": similarity_metrics,
            "similarity_score_mean": np.round(
                np.mean(reynolds_to_brownian_similarity), 2
            ),
            "similarity_score_std": np.round(
                np.std(reynolds_to_brownian_similarity), 2
            ),
        },
    )
    df = append_row_to_dataframe(
        df,
        {
            "similarity_type": "Vicsek to Vicsek",
            "metrics": similarity_metrics,
            "similarity_score_mean": np.round(np.mean(vicsek_to_vicsek_similarity), 2),
            "similarity_score_std": np.round(np.std(vicsek_to_vicsek_similarity), 2),
        },
    )
    df = append_row_to_dataframe(
        df,
        {
            "similarity_type": "Vicsek to Aggregation",
            "metrics": similarity_metrics,
            "similarity_score_mean": np.round(
                np.mean(vicsek_to_aggregation_similarity), 2
            ),
            "similarity_score_std": np.round(
                np.std(vicsek_to_aggregation_similarity), 2
            ),
        },
    )
    df = append_row_to_dataframe(
        df,
        {
            "similarity_type": "Vicsek to Dispersion",
            "metrics": similarity_metrics,
            "similarity_score_mean": np.round(
                np.mean(vicsek_to_dispersion_similarity), 2
            ),
            "similarity_score_std": np.round(
                np.std(vicsek_to_dispersion_similarity), 2
            ),
        },
    )
    df = append_row_to_dataframe(
        df,
        {
            "similarity_type": "Vicsek to Ballistic",
            "metrics": similarity_metrics,
            "similarity_score_mean": np.round(
                np.mean(vicsek_to_ballistic_similarity), 2
            ),
            "similarity_score_std": np.round(np.std(vicsek_to_ballistic_similarity), 2),
        },
    )
    df = append_row_to_dataframe(
        df,
        {
            "similarity_type": "Vicsek to Brownian",
            "metrics": similarity_metrics,
            "similarity_score_mean": np.round(
                np.mean(vicsek_to_brownian_similarity), 2
            ),
            "similarity_score_std": np.round(np.std(vicsek_to_brownian_similarity), 2),
        },
    )
    df = append_row_to_dataframe(
        df,
        {
            "similarity_type": "Aggregation to Aggregation",
            "metrics": similarity_metrics,
            "similarity_score_mean": np.round(
                np.mean(aggregation_to_aggregation_similarity), 2
            ),
            "similarity_score_std": np.round(
                np.std(aggregation_to_aggregation_similarity), 2
            ),
        },
    )
    df = append_row_to_dataframe(
        df,
        {
            "similarity_type": "Aggregation to Dispersion",
            "metrics": similarity_metrics,
            "similarity_score_mean": np.round(
                np.mean(aggregation_to_dispersion_similarity), 2
            ),
            "similarity_score_std": np.round(
                np.std(aggregation_to_dispersion_similarity), 2
            ),
        },
    )
    df = append_row_to_dataframe(
        df,
        {
            "similarity_type": "Aggregation to Ballistic",
            "metrics": similarity_metrics,
            "similarity_score_mean": np.round(
                np.mean(aggregation_to_ballistic_similarity), 2
            ),
            "similarity_score_std": np.round(
                np.std(aggregation_to_ballistic_similarity), 2
            ),
        },
    )
    df = append_row_to_dataframe(
        df,
        {
            "similarity_type": "Aggregation to Brownian",
            "metrics": similarity_metrics,
            "similarity_score_mean": np.round(
                np.mean(aggregation_to_brownian_similarity), 2
            ),
            "similarity_score_std": np.round(
                np.std(aggregation_to_brownian_similarity), 2
            ),
        },
    )
    df = append_row_to_dataframe(
        df,
        {
            "similarity_type": "Dispersion to Dispersion",
            "metrics": similarity_metrics,
            "similarity_score_mean": np.round(
                np.mean(dispersion_to_dispersion_similarity), 2
            ),
            "similarity_score_std": np.round(
                np.std(dispersion_to_dispersion_similarity), 2
            ),
        },
    )
    df = append_row_to_dataframe(
        df,
        {
            "similarity_type": "Dispersion to Ballistic",
            "metrics": similarity_metrics,
            "similarity_score_mean": np.round(
                np.mean(dispersion_to_ballistic_similarity), 2
            ),
            "similarity_score_std": np.round(
                np.std(dispersion_to_ballistic_similarity), 2
            ),
        },
    )
    df = append_row_to_dataframe(
        df,
        {
            "similarity_type": "Dispersion to Brownian",
            "metrics": similarity_metrics,
            "similarity_score_mean": np.round(
                np.mean(dispersion_to_brownian_similarity), 2
            ),
            "similarity_score_std": np.round(
                np.std(dispersion_to_brownian_similarity), 2
            ),
        },
    )
    df = append_row_to_dataframe(
        df,
        {
            "similarity_type": "Ballistic to Ballistic",
            "metrics": similarity_metrics,
            "similarity_score_mean": np.round(
                np.mean(ballistic_to_ballistic_similarity), 2
            ),
            "similarity_score_std": np.round(
                np.std(ballistic_to_ballistic_similarity), 2
            ),
        },
    )
    df = append_row_to_dataframe(
        df,
        {
            "similarity_type": "Ballistic to Brownian",
            "metrics": similarity_metrics,
            "similarity_score_mean": np.round(
                np.mean(ballistic_to_brownian_similarity), 2
            ),
            "similarity_score_std": np.round(
                np.std(ballistic_to_brownian_similarity), 2
            ),
        },
    )
    df = append_row_to_dataframe(
        df,
        {
            "similarity_type": "Brownian to Brownian",
            "metrics": similarity_metrics,
            "similarity_score_mean": np.round(
                np.mean(brownian_to_brownian_similarity), 2
            ),
            "similarity_score_std": np.round(
                np.std(brownian_to_brownian_similarity), 2
            ),
        },
    )
    export_dataframe(df, csv_path_name, latex_columns=None)

    log.info("\n" + 20 * "=" + "\n")


def get_combined_state_count_sim_behaviours(
    reynolds_40b: np.ndarray,
    reynolds_30b: np.ndarray,
    reynolds_40u: np.ndarray,
    vicsek_40b: np.ndarray,
    vicsek_30b: np.ndarray,
    vicsek_40u: np.ndarray,
    aggregation_40b: np.ndarray,
    aggregation_30b: np.ndarray,
    aggregation_40u: np.ndarray,
    dispersion_40b: np.ndarray,
    dispersion_30b: np.ndarray,
    dispersion_40u: np.ndarray,
    ballistic_40b: np.ndarray,
    ballistic_30b: np.ndarray,
    ballistic_40u: np.ndarray,
    brownian_40b: np.ndarray,
    brownian_30b: np.ndarray,
    brownian_40u: np.ndarray,
    similarity_metrics: tuple,
    csv_path_name: str,
    cfg: DictConfig,
):
    """"""
    log.info("\n" + 20 * "=" + "\n")

    reynolds_to_reynolds_similarity = np.array(
        [
            compute_combined_state_count_measure(
                reynolds_40b,
                reynolds_40b,
                cfg.metrics.combined_state_count.num_bins,
                cfg.metrics.combined_state_count.cut_off,
            )[2],
            compute_combined_state_count_measure(
                reynolds_30b,
                reynolds_30b,
                cfg.metrics.combined_state_count.num_bins,
                cfg.metrics.combined_state_count.cut_off,
            )[2],
            compute_combined_state_count_measure(
                reynolds_40u,
                reynolds_40u,
                cfg.metrics.combined_state_count.num_bins,
                cfg.metrics.combined_state_count.cut_off,
            )[2],
        ]
    )
    reynolds_to_vicsek_similarity = np.array(
        [
            compute_combined_state_count_measure(
                reynolds_40b,
                vicsek_40b,
                cfg.metrics.combined_state_count.num_bins,
                cfg.metrics.combined_state_count.cut_off,
            )[2],
            compute_combined_state_count_measure(
                reynolds_30b,
                vicsek_30b,
                cfg.metrics.combined_state_count.num_bins,
                cfg.metrics.combined_state_count.cut_off,
            )[2],
            compute_combined_state_count_measure(
                reynolds_40u,
                vicsek_40u,
                cfg.metrics.combined_state_count.num_bins,
                cfg.metrics.combined_state_count.cut_off,
            )[2],
        ]
    )
    reynolds_to_aggregation_similarity = np.array(
        [
            compute_combined_state_count_measure(
                reynolds_40b,
                aggregation_40b,
                cfg.metrics.combined_state_count.num_bins,
                cfg.metrics.combined_state_count.cut_off,
            )[2],
            compute_combined_state_count_measure(
                reynolds_30b,
                aggregation_30b,
                cfg.metrics.combined_state_count.num_bins,
                cfg.metrics.combined_state_count.cut_off,
            )[2],
            compute_combined_state_count_measure(
                reynolds_40u,
                aggregation_40u,
                cfg.metrics.combined_state_count.num_bins,
                cfg.metrics.combined_state_count.cut_off,
            )[2],
        ]
    )
    reynolds_to_dispersion_similarity = np.array(
        [
            compute_combined_state_count_measure(
                reynolds_40b,
                dispersion_40b,
                cfg.metrics.combined_state_count.num_bins,
                cfg.metrics.combined_state_count.cut_off,
            )[2],
            compute_combined_state_count_measure(
                reynolds_30b,
                dispersion_30b,
                cfg.metrics.combined_state_count.num_bins,
                cfg.metrics.combined_state_count.cut_off,
            )[2],
            compute_combined_state_count_measure(
                reynolds_40u,
                dispersion_40u,
                cfg.metrics.combined_state_count.num_bins,
                cfg.metrics.combined_state_count.cut_off,
            )[2],
        ]
    )
    reynolds_to_ballistic_similarity = np.array(
        [
            compute_combined_state_count_measure(
                reynolds_40b,
                ballistic_40b,
                cfg.metrics.combined_state_count.num_bins,
                cfg.metrics.combined_state_count.cut_off,
            )[2],
            compute_combined_state_count_measure(
                reynolds_30b,
                ballistic_30b,
                cfg.metrics.combined_state_count.num_bins,
                cfg.metrics.combined_state_count.cut_off,
            )[2],
            compute_combined_state_count_measure(
                reynolds_40u,
                ballistic_40u,
                cfg.metrics.combined_state_count.num_bins,
                cfg.metrics.combined_state_count.cut_off,
            )[2],
        ]
    )
    reynolds_to_brownian_similarity = np.array(
        [
            compute_combined_state_count_measure(
                reynolds_40b,
                brownian_40b,
                cfg.metrics.combined_state_count.num_bins,
                cfg.metrics.combined_state_count.cut_off,
            )[2],
            compute_combined_state_count_measure(
                reynolds_30b,
                brownian_30b,
                cfg.metrics.combined_state_count.num_bins,
                cfg.metrics.combined_state_count.cut_off,
            )[2],
            compute_combined_state_count_measure(
                reynolds_40u,
                brownian_40u,
                cfg.metrics.combined_state_count.num_bins,
                cfg.metrics.combined_state_count.cut_off,
            )[2],
        ]
    )
    vicsek_to_vicsek_similarity = np.array(
        [
            compute_combined_state_count_measure(
                vicsek_40b,
                vicsek_40b,
                cfg.metrics.combined_state_count.num_bins,
                cfg.metrics.combined_state_count.cut_off,
            )[2],
            compute_combined_state_count_measure(
                vicsek_30b,
                vicsek_30b,
                cfg.metrics.combined_state_count.num_bins,
                cfg.metrics.combined_state_count.cut_off,
            )[2],
            compute_combined_state_count_measure(
                vicsek_40u,
                vicsek_40u,
                cfg.metrics.combined_state_count.num_bins,
                cfg.metrics.combined_state_count.cut_off,
            )[2],
        ]
    )
    vicsek_to_aggregation_similarity = np.array(
        [
            compute_combined_state_count_measure(
                vicsek_40b,
                aggregation_40b,
                cfg.metrics.combined_state_count.num_bins,
                cfg.metrics.combined_state_count.cut_off,
            )[2],
            compute_combined_state_count_measure(
                vicsek_30b,
                aggregation_30b,
                cfg.metrics.combined_state_count.num_bins,
                cfg.metrics.combined_state_count.cut_off,
            )[2],
            compute_combined_state_count_measure(
                vicsek_40u,
                aggregation_40u,
                cfg.metrics.combined_state_count.num_bins,
                cfg.metrics.combined_state_count.cut_off,
            )[2],
        ]
    )
    vicsek_to_dispersion_similarity = np.array(
        [
            compute_combined_state_count_measure(
                vicsek_40b,
                dispersion_40b,
                cfg.metrics.combined_state_count.num_bins,
                cfg.metrics.combined_state_count.cut_off,
            )[2],
            compute_combined_state_count_measure(
                vicsek_30b,
                dispersion_30b,
                cfg.metrics.combined_state_count.num_bins,
                cfg.metrics.combined_state_count.cut_off,
            )[2],
            compute_combined_state_count_measure(
                vicsek_40u,
                dispersion_40u,
                cfg.metrics.combined_state_count.num_bins,
                cfg.metrics.combined_state_count.cut_off,
            )[2],
        ]
    )
    vicsek_to_ballistic_similarity = np.array(
        [
            compute_combined_state_count_measure(
                vicsek_40b,
                ballistic_40b,
                cfg.metrics.combined_state_count.num_bins,
                cfg.metrics.combined_state_count.cut_off,
            )[2],
            compute_combined_state_count_measure(
                vicsek_30b,
                ballistic_30b,
                cfg.metrics.combined_state_count.num_bins,
                cfg.metrics.combined_state_count.cut_off,
            )[2],
            compute_combined_state_count_measure(
                vicsek_40u,
                ballistic_40u,
                cfg.metrics.combined_state_count.num_bins,
                cfg.metrics.combined_state_count.cut_off,
            )[2],
        ]
    )
    vicsek_to_brownian_similarity = np.array(
        [
            compute_combined_state_count_measure(
                vicsek_40b,
                brownian_40b,
                cfg.metrics.combined_state_count.num_bins,
                cfg.metrics.combined_state_count.cut_off,
            )[2],
            compute_combined_state_count_measure(
                vicsek_30b,
                brownian_30b,
                cfg.metrics.combined_state_count.num_bins,
                cfg.metrics.combined_state_count.cut_off,
            )[2],
            compute_combined_state_count_measure(
                vicsek_40u,
                brownian_40u,
                cfg.metrics.combined_state_count.num_bins,
                cfg.metrics.combined_state_count.cut_off,
            )[2],
        ]
    )
    aggregation_to_aggregation_similarity = np.array(
        [
            compute_combined_state_count_measure(
                aggregation_40b,
                aggregation_40b,
                cfg.metrics.combined_state_count.num_bins,
                cfg.metrics.combined_state_count.cut_off,
            )[2],
            compute_combined_state_count_measure(
                aggregation_30b,
                aggregation_30b,
                cfg.metrics.combined_state_count.num_bins,
                cfg.metrics.combined_state_count.cut_off,
            )[2],
            compute_combined_state_count_measure(
                aggregation_40u,
                aggregation_40u,
                cfg.metrics.combined_state_count.num_bins,
                cfg.metrics.combined_state_count.cut_off,
            )[2],
        ]
    )
    aggregation_to_dispersion_similarity = np.array(
        [
            compute_combined_state_count_measure(
                aggregation_40b,
                dispersion_40b,
                cfg.metrics.combined_state_count.num_bins,
                cfg.metrics.combined_state_count.cut_off,
            )[2],
            compute_combined_state_count_measure(
                aggregation_30b,
                dispersion_30b,
                cfg.metrics.combined_state_count.num_bins,
                cfg.metrics.combined_state_count.cut_off,
            )[2],
            compute_combined_state_count_measure(
                aggregation_40u,
                dispersion_40u,
                cfg.metrics.combined_state_count.num_bins,
                cfg.metrics.combined_state_count.cut_off,
            )[2],
        ]
    )
    aggregation_to_ballistic_similarity = np.array(
        [
            compute_combined_state_count_measure(
                aggregation_40b,
                ballistic_40b,
                cfg.metrics.combined_state_count.num_bins,
                cfg.metrics.combined_state_count.cut_off,
            )[2],
            compute_combined_state_count_measure(
                aggregation_30b,
                ballistic_30b,
                cfg.metrics.combined_state_count.num_bins,
                cfg.metrics.combined_state_count.cut_off,
            )[2],
            compute_combined_state_count_measure(
                aggregation_40u,
                ballistic_40u,
                cfg.metrics.combined_state_count.num_bins,
                cfg.metrics.combined_state_count.cut_off,
            )[2],
        ]
    )
    aggregation_to_brownian_similarity = np.array(
        [
            compute_combined_state_count_measure(
                aggregation_40b,
                brownian_40b,
                cfg.metrics.combined_state_count.num_bins,
                cfg.metrics.combined_state_count.cut_off,
            )[2],
            compute_combined_state_count_measure(
                aggregation_30b,
                brownian_30b,
                cfg.metrics.combined_state_count.num_bins,
                cfg.metrics.combined_state_count.cut_off,
            )[2],
            compute_combined_state_count_measure(
                aggregation_40u,
                brownian_40u,
                cfg.metrics.combined_state_count.num_bins,
                cfg.metrics.combined_state_count.cut_off,
            )[2],
        ]
    )
    dispersion_to_dispersion_similarity = np.array(
        [
            compute_combined_state_count_measure(
                dispersion_40b,
                dispersion_40b,
                cfg.metrics.combined_state_count.num_bins,
                cfg.metrics.combined_state_count.cut_off,
            )[2],
            compute_combined_state_count_measure(
                dispersion_30b,
                dispersion_30b,
                cfg.metrics.combined_state_count.num_bins,
                cfg.metrics.combined_state_count.cut_off,
            )[2],
            compute_combined_state_count_measure(
                dispersion_40u,
                dispersion_40u,
                cfg.metrics.combined_state_count.num_bins,
                cfg.metrics.combined_state_count.cut_off,
            )[2],
        ]
    )
    dispersion_to_ballistic_similarity = np.array(
        [
            compute_combined_state_count_measure(
                dispersion_40b,
                ballistic_40b,
                cfg.metrics.combined_state_count.num_bins,
                cfg.metrics.combined_state_count.cut_off,
            )[2],
            compute_combined_state_count_measure(
                dispersion_30b,
                ballistic_30b,
                cfg.metrics.combined_state_count.num_bins,
                cfg.metrics.combined_state_count.cut_off,
            )[2],
            compute_combined_state_count_measure(
                dispersion_40u,
                ballistic_40u,
                cfg.metrics.combined_state_count.num_bins,
                cfg.metrics.combined_state_count.cut_off,
            )[2],
        ]
    )
    dispersion_to_brownian_similarity = np.array(
        [
            compute_combined_state_count_measure(
                dispersion_40b,
                brownian_40b,
                cfg.metrics.combined_state_count.num_bins,
                cfg.metrics.combined_state_count.cut_off,
            )[2],
            compute_combined_state_count_measure(
                dispersion_30b,
                brownian_30b,
                cfg.metrics.combined_state_count.num_bins,
                cfg.metrics.combined_state_count.cut_off,
            )[2],
            compute_combined_state_count_measure(
                dispersion_40u,
                brownian_40u,
                cfg.metrics.combined_state_count.num_bins,
                cfg.metrics.combined_state_count.cut_off,
            )[2],
        ]
    )
    ballistic_to_ballistic_similarity = np.array(
        [
            compute_combined_state_count_measure(
                ballistic_40b,
                ballistic_40b,
                cfg.metrics.combined_state_count.num_bins,
                cfg.metrics.combined_state_count.cut_off,
            )[2],
            compute_combined_state_count_measure(
                ballistic_30b,
                ballistic_30b,
                cfg.metrics.combined_state_count.num_bins,
                cfg.metrics.combined_state_count.cut_off,
            )[2],
            compute_combined_state_count_measure(
                ballistic_40u,
                ballistic_40u,
                cfg.metrics.combined_state_count.num_bins,
                cfg.metrics.combined_state_count.cut_off,
            )[2],
        ]
    )
    ballistic_to_brownian_similarity = np.array(
        [
            compute_combined_state_count_measure(
                ballistic_40b,
                brownian_40b,
                cfg.metrics.combined_state_count.num_bins,
                cfg.metrics.combined_state_count.cut_off,
            )[2],
            compute_combined_state_count_measure(
                ballistic_30b,
                brownian_30b,
                cfg.metrics.combined_state_count.num_bins,
                cfg.metrics.combined_state_count.cut_off,
            )[2],
            compute_combined_state_count_measure(
                ballistic_40u,
                brownian_40u,
                cfg.metrics.combined_state_count.num_bins,
                cfg.metrics.combined_state_count.cut_off,
            )[2],
        ]
    )
    brownian_to_brownian_similarity = np.array(
        [
            compute_combined_state_count_measure(
                brownian_40b,
                brownian_40b,
                cfg.metrics.combined_state_count.num_bins,
                cfg.metrics.combined_state_count.cut_off,
            )[2],
            compute_combined_state_count_measure(
                brownian_30b,
                brownian_30b,
                cfg.metrics.combined_state_count.num_bins,
                cfg.metrics.combined_state_count.cut_off,
            )[2],
            compute_combined_state_count_measure(
                brownian_40u,
                brownian_40u,
                cfg.metrics.combined_state_count.num_bins,
                cfg.metrics.combined_state_count.cut_off,
            )[2],
        ]
    )

    df = open_dataframe(
        csv_path_name,
        [
            "similarity_type",
            "metrics",
            "similarity_score_mean",
            "similarity_score_std",
        ],
    )
    df = append_row_to_dataframe(
        df,
        {
            "similarity_type": "Reynolds to Reynolds",
            "metrics": similarity_metrics,
            "similarity_score_mean": np.round(
                np.mean(reynolds_to_reynolds_similarity), 2
            ),
            "similarity_score_std": np.round(
                np.std(reynolds_to_reynolds_similarity), 2
            ),
        },
    )
    df = append_row_to_dataframe(
        df,
        {
            "similarity_type": "Reynolds to Vicsek",
            "metrics": similarity_metrics,
            "similarity_score_mean": np.round(
                np.mean(reynolds_to_vicsek_similarity), 2
            ),
            "similarity_score_std": np.round(np.std(reynolds_to_vicsek_similarity), 2),
        },
    )
    df = append_row_to_dataframe(
        df,
        {
            "similarity_type": "Reynolds to Aggregation",
            "metrics": similarity_metrics,
            "similarity_score_mean": np.round(
                np.mean(reynolds_to_aggregation_similarity), 2
            ),
            "similarity_score_std": np.round(
                np.std(reynolds_to_aggregation_similarity), 2
            ),
        },
    )
    df = append_row_to_dataframe(
        df,
        {
            "similarity_type": "Reynolds to Dispersion",
            "metrics": similarity_metrics,
            "similarity_score_mean": np.round(
                np.mean(reynolds_to_dispersion_similarity), 2
            ),
            "similarity_score_std": np.round(
                np.std(reynolds_to_dispersion_similarity), 2
            ),
        },
    )
    df = append_row_to_dataframe(
        df,
        {
            "similarity_type": "Reynolds to Ballistic",
            "metrics": similarity_metrics,
            "similarity_score_mean": np.round(
                np.mean(reynolds_to_ballistic_similarity), 2
            ),
            "similarity_score_std": np.round(
                np.std(reynolds_to_ballistic_similarity), 2
            ),
        },
    )
    df = append_row_to_dataframe(
        df,
        {
            "similarity_type": "Reynolds to Brownian",
            "metrics": similarity_metrics,
            "similarity_score_mean": np.round(
                np.mean(reynolds_to_brownian_similarity), 2
            ),
            "similarity_score_std": np.round(
                np.std(reynolds_to_brownian_similarity), 2
            ),
        },
    )
    df = append_row_to_dataframe(
        df,
        {
            "similarity_type": "Vicsek to Vicsek",
            "metrics": similarity_metrics,
            "similarity_score_mean": np.round(np.mean(vicsek_to_vicsek_similarity), 2),
            "similarity_score_std": np.round(np.std(vicsek_to_vicsek_similarity), 2),
        },
    )
    df = append_row_to_dataframe(
        df,
        {
            "similarity_type": "Vicsek to Aggregation",
            "metrics": similarity_metrics,
            "similarity_score_mean": np.round(
                np.mean(vicsek_to_aggregation_similarity), 2
            ),
            "similarity_score_std": np.round(
                np.std(vicsek_to_aggregation_similarity), 2
            ),
        },
    )
    df = append_row_to_dataframe(
        df,
        {
            "similarity_type": "Vicsek to Dispersion",
            "metrics": similarity_metrics,
            "similarity_score_mean": np.round(
                np.mean(vicsek_to_dispersion_similarity), 2
            ),
            "similarity_score_std": np.round(
                np.std(vicsek_to_dispersion_similarity), 2
            ),
        },
    )
    df = append_row_to_dataframe(
        df,
        {
            "similarity_type": "Vicsek to Ballistic",
            "metrics": similarity_metrics,
            "similarity_score_mean": np.round(
                np.mean(vicsek_to_ballistic_similarity), 2
            ),
            "similarity_score_std": np.round(np.std(vicsek_to_ballistic_similarity), 2),
        },
    )
    df = append_row_to_dataframe(
        df,
        {
            "similarity_type": "Vicsek to Brownian",
            "metrics": similarity_metrics,
            "similarity_score_mean": np.round(
                np.mean(vicsek_to_brownian_similarity), 2
            ),
            "similarity_score_std": np.round(np.std(vicsek_to_brownian_similarity), 2),
        },
    )
    df = append_row_to_dataframe(
        df,
        {
            "similarity_type": "Aggregation to Aggregation",
            "metrics": similarity_metrics,
            "similarity_score_mean": np.round(
                np.mean(aggregation_to_aggregation_similarity), 2
            ),
            "similarity_score_std": np.round(
                np.std(aggregation_to_aggregation_similarity), 2
            ),
        },
    )
    df = append_row_to_dataframe(
        df,
        {
            "similarity_type": "Aggregation to Dispersion",
            "metrics": similarity_metrics,
            "similarity_score_mean": np.round(
                np.mean(aggregation_to_dispersion_similarity), 2
            ),
            "similarity_score_std": np.round(
                np.std(aggregation_to_dispersion_similarity), 2
            ),
        },
    )
    df = append_row_to_dataframe(
        df,
        {
            "similarity_type": "Aggregation to Ballistic",
            "metrics": similarity_metrics,
            "similarity_score_mean": np.round(
                np.mean(aggregation_to_ballistic_similarity), 2
            ),
            "similarity_score_std": np.round(
                np.std(aggregation_to_ballistic_similarity), 2
            ),
        },
    )
    df = append_row_to_dataframe(
        df,
        {
            "similarity_type": "Aggregation to Brownian",
            "metrics": similarity_metrics,
            "similarity_score_mean": np.round(
                np.mean(aggregation_to_brownian_similarity), 2
            ),
            "similarity_score_std": np.round(
                np.std(aggregation_to_brownian_similarity), 2
            ),
        },
    )
    df = append_row_to_dataframe(
        df,
        {
            "similarity_type": "Dispersion to Dispersion",
            "metrics": similarity_metrics,
            "similarity_score_mean": np.round(
                np.mean(dispersion_to_dispersion_similarity), 2
            ),
            "similarity_score_std": np.round(
                np.std(dispersion_to_dispersion_similarity), 2
            ),
        },
    )
    df = append_row_to_dataframe(
        df,
        {
            "similarity_type": "Dispersion to Ballistic",
            "metrics": similarity_metrics,
            "similarity_score_mean": np.round(
                np.mean(dispersion_to_ballistic_similarity), 2
            ),
            "similarity_score_std": np.round(
                np.std(dispersion_to_ballistic_similarity), 2
            ),
        },
    )
    df = append_row_to_dataframe(
        df,
        {
            "similarity_type": "Dispersion to Brownian",
            "metrics": similarity_metrics,
            "similarity_score_mean": np.round(
                np.mean(dispersion_to_brownian_similarity), 2
            ),
            "similarity_score_std": np.round(
                np.std(dispersion_to_brownian_similarity), 2
            ),
        },
    )
    df = append_row_to_dataframe(
        df,
        {
            "similarity_type": "Ballistic to Ballistic",
            "metrics": similarity_metrics,
            "similarity_score_mean": np.round(
                np.mean(ballistic_to_ballistic_similarity), 2
            ),
            "similarity_score_std": np.round(
                np.std(ballistic_to_ballistic_similarity), 2
            ),
        },
    )
    df = append_row_to_dataframe(
        df,
        {
            "similarity_type": "Ballistic to Brownian",
            "metrics": similarity_metrics,
            "similarity_score_mean": np.round(
                np.mean(ballistic_to_brownian_similarity), 2
            ),
            "similarity_score_std": np.round(
                np.std(ballistic_to_brownian_similarity), 2
            ),
        },
    )
    df = append_row_to_dataframe(
        df,
        {
            "similarity_type": "Brownian to Brownian",
            "metrics": similarity_metrics,
            "similarity_score_mean": np.round(
                np.mean(brownian_to_brownian_similarity), 2
            ),
            "similarity_score_std": np.round(
                np.std(brownian_to_brownian_similarity), 2
            ),
        },
    )
    export_dataframe(df, csv_path_name, latex_columns=None)

    log.info("\n" + 20 * "=" + "\n")


def get_sampled_average_state_sim_behaviours(
    reynolds_40b: np.ndarray,
    reynolds_30b: np.ndarray,
    reynolds_40u: np.ndarray,
    vicsek_40b: np.ndarray,
    vicsek_30b: np.ndarray,
    vicsek_40u: np.ndarray,
    aggregation_40b: np.ndarray,
    aggregation_30b: np.ndarray,
    aggregation_40u: np.ndarray,
    dispersion_40b: np.ndarray,
    dispersion_30b: np.ndarray,
    dispersion_40u: np.ndarray,
    ballistic_40b: np.ndarray,
    ballistic_30b: np.ndarray,
    ballistic_40u: np.ndarray,
    brownian_40b: np.ndarray,
    brownian_30b: np.ndarray,
    brownian_40u: np.ndarray,
    similarity_metrics: tuple,
    csv_path_name: str,
    num_windows: int,
    cfg: DictConfig,
):
    """"""
    log.info("\n" + 20 * "=" + "\n")

    reynolds_to_reynolds_similarity = np.array(
        [
            compute_sampled_average_state_measure(
                reynolds_40b,
                reynolds_40b,
                num_time_windows=num_windows,
                num_steps_sim=reynolds_40b.shape[1],
                eps=cfg.metrics.sampled_average_state.eps,
            ),
            compute_sampled_average_state_measure(
                reynolds_30b,
                reynolds_30b,
                num_time_windows=num_windows,
                num_steps_sim=reynolds_30b.shape[1],
                eps=cfg.metrics.sampled_average_state.eps,
            ),
            compute_sampled_average_state_measure(
                reynolds_40u,
                reynolds_40u,
                num_time_windows=num_windows,
                num_steps_sim=reynolds_40u.shape[1],
                eps=cfg.metrics.sampled_average_state.eps,
            ),
        ]
    )
    reynolds_to_vicsek_similarity = np.array(
        [
            compute_sampled_average_state_measure(
                reynolds_40b,
                vicsek_40b,
                num_time_windows=num_windows,
                num_steps_sim=reynolds_40b.shape[1],
                eps=cfg.metrics.sampled_average_state.eps,
            ),
            compute_sampled_average_state_measure(
                reynolds_30b,
                vicsek_30b,
                num_time_windows=num_windows,
                num_steps_sim=reynolds_30b.shape[1],
                eps=cfg.metrics.sampled_average_state.eps,
            ),
            compute_sampled_average_state_measure(
                reynolds_40u,
                vicsek_40u,
                num_time_windows=num_windows,
                num_steps_sim=reynolds_40u.shape[1],
                eps=cfg.metrics.sampled_average_state.eps,
            ),
        ]
    )
    reynolds_to_aggregation_similarity = np.array(
        [
            compute_sampled_average_state_measure(
                reynolds_40b,
                aggregation_40b,
                num_time_windows=num_windows,
                num_steps_sim=reynolds_40b.shape[1],
                eps=cfg.metrics.sampled_average_state.eps,
            ),
            compute_sampled_average_state_measure(
                reynolds_30b,
                aggregation_30b,
                num_time_windows=num_windows,
                num_steps_sim=reynolds_30b.shape[1],
                eps=cfg.metrics.sampled_average_state.eps,
            ),
            compute_sampled_average_state_measure(
                reynolds_40u,
                aggregation_40u,
                num_time_windows=num_windows,
                num_steps_sim=reynolds_40u.shape[1],
                eps=cfg.metrics.sampled_average_state.eps,
            ),
        ]
    )
    reynolds_to_dispersion_similarity = np.array(
        [
            compute_sampled_average_state_measure(
                reynolds_40b,
                dispersion_40b,
                num_time_windows=num_windows,
                num_steps_sim=reynolds_40b.shape[1],
                eps=cfg.metrics.sampled_average_state.eps,
            ),
            compute_sampled_average_state_measure(
                reynolds_30b,
                dispersion_30b,
                num_time_windows=num_windows,
                num_steps_sim=reynolds_30b.shape[1],
                eps=cfg.metrics.sampled_average_state.eps,
            ),
            compute_sampled_average_state_measure(
                reynolds_40u,
                dispersion_40u,
                num_time_windows=num_windows,
                num_steps_sim=reynolds_40u.shape[1],
                eps=cfg.metrics.sampled_average_state.eps,
            ),
        ]
    )
    reynolds_to_ballistic_similarity = np.array(
        [
            compute_sampled_average_state_measure(
                reynolds_40b,
                ballistic_40b,
                num_time_windows=num_windows,
                num_steps_sim=reynolds_40b.shape[1],
                eps=cfg.metrics.sampled_average_state.eps,
            ),
            compute_sampled_average_state_measure(
                reynolds_30b,
                ballistic_30b,
                num_time_windows=num_windows,
                num_steps_sim=reynolds_30b.shape[1],
                eps=cfg.metrics.sampled_average_state.eps,
            ),
            compute_sampled_average_state_measure(
                reynolds_40u,
                ballistic_40u,
                num_time_windows=num_windows,
                num_steps_sim=reynolds_40u.shape[1],
                eps=cfg.metrics.sampled_average_state.eps,
            ),
        ]
    )
    reynolds_to_brownian_similarity = np.array(
        [
            compute_sampled_average_state_measure(
                reynolds_40b,
                brownian_40b,
                num_time_windows=num_windows,
                num_steps_sim=reynolds_40b.shape[1],
                eps=cfg.metrics.sampled_average_state.eps,
            ),
            compute_sampled_average_state_measure(
                reynolds_30b,
                brownian_30b,
                num_time_windows=num_windows,
                num_steps_sim=reynolds_30b.shape[1],
                eps=cfg.metrics.sampled_average_state.eps,
            ),
            compute_sampled_average_state_measure(
                reynolds_40u,
                brownian_40u,
                num_time_windows=num_windows,
                num_steps_sim=reynolds_40u.shape[1],
                eps=cfg.metrics.sampled_average_state.eps,
            ),
        ]
    )
    vicsek_to_vicsek_similarity = np.array(
        [
            compute_sampled_average_state_measure(
                vicsek_40b,
                vicsek_40b,
                num_time_windows=num_windows,
                num_steps_sim=vicsek_40b.shape[1],
                eps=cfg.metrics.sampled_average_state.eps,
            ),
            compute_sampled_average_state_measure(
                vicsek_30b,
                vicsek_30b,
                num_time_windows=num_windows,
                num_steps_sim=vicsek_30b.shape[1],
                eps=cfg.metrics.sampled_average_state.eps,
            ),
            compute_sampled_average_state_measure(
                vicsek_40u,
                vicsek_40u,
                num_time_windows=num_windows,
                num_steps_sim=vicsek_40u.shape[1],
                eps=cfg.metrics.sampled_average_state.eps,
            ),
        ]
    )
    vicsek_to_aggregation_similarity = np.array(
        [
            compute_sampled_average_state_measure(
                vicsek_40b,
                aggregation_40b,
                num_time_windows=num_windows,
                num_steps_sim=vicsek_40b.shape[1],
                eps=cfg.metrics.sampled_average_state.eps,
            ),
            compute_sampled_average_state_measure(
                vicsek_30b,
                aggregation_30b,
                num_time_windows=num_windows,
                num_steps_sim=vicsek_30b.shape[1],
                eps=cfg.metrics.sampled_average_state.eps,
            ),
            compute_sampled_average_state_measure(
                vicsek_40u,
                aggregation_40u,
                num_time_windows=num_windows,
                num_steps_sim=vicsek_40u.shape[1],
                eps=cfg.metrics.sampled_average_state.eps,
            ),
        ]
    )
    vicsek_to_dispersion_similarity = np.array(
        [
            compute_sampled_average_state_measure(
                vicsek_40b,
                dispersion_40b,
                num_time_windows=num_windows,
                num_steps_sim=vicsek_40b.shape[1],
                eps=cfg.metrics.sampled_average_state.eps,
            ),
            compute_sampled_average_state_measure(
                vicsek_30b,
                dispersion_30b,
                num_time_windows=num_windows,
                num_steps_sim=vicsek_30b.shape[1],
                eps=cfg.metrics.sampled_average_state.eps,
            ),
            compute_sampled_average_state_measure(
                vicsek_40u,
                dispersion_40u,
                num_time_windows=num_windows,
                num_steps_sim=vicsek_40u.shape[1],
                eps=cfg.metrics.sampled_average_state.eps,
            ),
        ]
    )
    vicsek_to_ballistic_similarity = np.array(
        [
            compute_sampled_average_state_measure(
                vicsek_40b,
                ballistic_40b,
                num_time_windows=num_windows,
                num_steps_sim=vicsek_40b.shape[1],
                eps=cfg.metrics.sampled_average_state.eps,
            ),
            compute_sampled_average_state_measure(
                vicsek_30b,
                ballistic_30b,
                num_time_windows=num_windows,
                num_steps_sim=vicsek_30b.shape[1],
                eps=cfg.metrics.sampled_average_state.eps,
            ),
            compute_sampled_average_state_measure(
                vicsek_40u,
                ballistic_40u,
                num_time_windows=num_windows,
                num_steps_sim=vicsek_40u.shape[1],
                eps=cfg.metrics.sampled_average_state.eps,
            ),
        ]
    )
    vicsek_to_brownian_similarity = np.array(
        [
            compute_sampled_average_state_measure(
                vicsek_40b,
                brownian_40b,
                num_time_windows=num_windows,
                num_steps_sim=vicsek_40b.shape[1],
                eps=cfg.metrics.sampled_average_state.eps,
            ),
            compute_sampled_average_state_measure(
                vicsek_30b,
                brownian_30b,
                num_time_windows=num_windows,
                num_steps_sim=vicsek_30b.shape[1],
                eps=cfg.metrics.sampled_average_state.eps,
            ),
            compute_sampled_average_state_measure(
                vicsek_40u,
                brownian_40u,
                num_time_windows=num_windows,
                num_steps_sim=vicsek_40u.shape[1],
                eps=cfg.metrics.sampled_average_state.eps,
            ),
        ]
    )
    aggregation_to_aggregation_similarity = np.array(
        [
            compute_sampled_average_state_measure(
                aggregation_40b,
                aggregation_40b,
                num_time_windows=num_windows,
                num_steps_sim=aggregation_40b.shape[1],
                eps=cfg.metrics.sampled_average_state.eps,
            ),
            compute_sampled_average_state_measure(
                aggregation_30b,
                aggregation_30b,
                num_time_windows=num_windows,
                num_steps_sim=aggregation_30b.shape[1],
                eps=cfg.metrics.sampled_average_state.eps,
            ),
            compute_sampled_average_state_measure(
                aggregation_40u,
                aggregation_40u,
                num_time_windows=num_windows,
                num_steps_sim=aggregation_40u.shape[1],
                eps=cfg.metrics.sampled_average_state.eps,
            ),
        ]
    )
    aggregation_to_dispersion_similarity = np.array(
        [
            compute_sampled_average_state_measure(
                aggregation_40b,
                dispersion_40b,
                num_time_windows=num_windows,
                num_steps_sim=aggregation_40b.shape[1],
                eps=cfg.metrics.sampled_average_state.eps,
            ),
            compute_sampled_average_state_measure(
                aggregation_30b,
                dispersion_30b,
                num_time_windows=num_windows,
                num_steps_sim=aggregation_30b.shape[1],
                eps=cfg.metrics.sampled_average_state.eps,
            ),
            compute_sampled_average_state_measure(
                aggregation_40u,
                dispersion_40u,
                num_time_windows=num_windows,
                num_steps_sim=aggregation_40u.shape[1],
                eps=cfg.metrics.sampled_average_state.eps,
            ),
        ]
    )
    aggregation_to_ballistic_similarity = np.array(
        [
            compute_sampled_average_state_measure(
                aggregation_40b,
                ballistic_40b,
                num_time_windows=num_windows,
                num_steps_sim=aggregation_40b.shape[1],
                eps=cfg.metrics.sampled_average_state.eps,
            ),
            compute_sampled_average_state_measure(
                aggregation_30b,
                ballistic_30b,
                num_time_windows=num_windows,
                num_steps_sim=aggregation_30b.shape[1],
                eps=cfg.metrics.sampled_average_state.eps,
            ),
            compute_sampled_average_state_measure(
                aggregation_40u,
                ballistic_40u,
                num_time_windows=num_windows,
                num_steps_sim=aggregation_40u.shape[1],
                eps=cfg.metrics.sampled_average_state.eps,
            ),
        ]
    )
    aggregation_to_brownian_similarity = np.array(
        [
            compute_sampled_average_state_measure(
                aggregation_40b,
                brownian_40b,
                num_time_windows=num_windows,
                num_steps_sim=aggregation_40b.shape[1],
                eps=cfg.metrics.sampled_average_state.eps,
            ),
            compute_sampled_average_state_measure(
                aggregation_30b,
                brownian_30b,
                num_time_windows=num_windows,
                num_steps_sim=aggregation_30b.shape[1],
                eps=cfg.metrics.sampled_average_state.eps,
            ),
            compute_sampled_average_state_measure(
                aggregation_40u,
                brownian_40u,
                num_time_windows=num_windows,
                num_steps_sim=aggregation_40u.shape[1],
                eps=cfg.metrics.sampled_average_state.eps,
            ),
        ]
    )
    dispersion_to_dispersion_similarity = np.array(
        [
            compute_sampled_average_state_measure(
                dispersion_40b,
                dispersion_40b,
                num_time_windows=num_windows,
                num_steps_sim=dispersion_40b.shape[1],
                eps=cfg.metrics.sampled_average_state.eps,
            ),
            compute_sampled_average_state_measure(
                dispersion_30b,
                dispersion_30b,
                num_time_windows=num_windows,
                num_steps_sim=dispersion_30b.shape[1],
                eps=cfg.metrics.sampled_average_state.eps,
            ),
            compute_sampled_average_state_measure(
                dispersion_40u,
                dispersion_40u,
                num_time_windows=num_windows,
                num_steps_sim=dispersion_40u.shape[1],
                eps=cfg.metrics.sampled_average_state.eps,
            ),
        ]
    )
    dispersion_to_ballistic_similarity = np.array(
        [
            compute_sampled_average_state_measure(
                dispersion_40b,
                ballistic_40b,
                num_time_windows=num_windows,
                num_steps_sim=dispersion_40b.shape[1],
                eps=cfg.metrics.sampled_average_state.eps,
            ),
            compute_sampled_average_state_measure(
                dispersion_30b,
                ballistic_30b,
                num_time_windows=num_windows,
                num_steps_sim=dispersion_30b.shape[1],
                eps=cfg.metrics.sampled_average_state.eps,
            ),
            compute_sampled_average_state_measure(
                dispersion_40u,
                ballistic_40u,
                num_time_windows=num_windows,
                num_steps_sim=dispersion_40u.shape[1],
                eps=cfg.metrics.sampled_average_state.eps,
            ),
        ]
    )
    dispersion_to_brownian_similarity = np.array(
        [
            compute_sampled_average_state_measure(
                dispersion_40b,
                brownian_40b,
                num_time_windows=num_windows,
                num_steps_sim=dispersion_40b.shape[1],
                eps=cfg.metrics.sampled_average_state.eps,
            ),
            compute_sampled_average_state_measure(
                dispersion_30b,
                brownian_30b,
                num_time_windows=num_windows,
                num_steps_sim=dispersion_30b.shape[1],
                eps=cfg.metrics.sampled_average_state.eps,
            ),
            compute_sampled_average_state_measure(
                dispersion_40u,
                brownian_40u,
                num_time_windows=num_windows,
                num_steps_sim=dispersion_40u.shape[1],
                eps=cfg.metrics.sampled_average_state.eps,
            ),
        ]
    )
    ballistic_to_ballistic_similarity = np.array(
        [
            compute_sampled_average_state_measure(
                ballistic_40b,
                ballistic_40b,
                num_time_windows=num_windows,
                num_steps_sim=ballistic_40b.shape[1],
                eps=cfg.metrics.sampled_average_state.eps,
            ),
            compute_sampled_average_state_measure(
                ballistic_30b,
                ballistic_30b,
                num_time_windows=num_windows,
                num_steps_sim=ballistic_30b.shape[1],
                eps=cfg.metrics.sampled_average_state.eps,
            ),
            compute_sampled_average_state_measure(
                ballistic_40u,
                ballistic_40u,
                num_time_windows=num_windows,
                num_steps_sim=ballistic_40u.shape[1],
                eps=cfg.metrics.sampled_average_state.eps,
            ),
        ]
    )
    ballistic_to_brownian_similarity = np.array(
        [
            compute_sampled_average_state_measure(
                ballistic_40b,
                brownian_40b,
                num_time_windows=num_windows,
                num_steps_sim=ballistic_40b.shape[1],
                eps=cfg.metrics.sampled_average_state.eps,
            ),
            compute_sampled_average_state_measure(
                ballistic_30b,
                brownian_30b,
                num_time_windows=num_windows,
                num_steps_sim=ballistic_30b.shape[1],
                eps=cfg.metrics.sampled_average_state.eps,
            ),
            compute_sampled_average_state_measure(
                ballistic_40u,
                brownian_40u,
                num_time_windows=num_windows,
                num_steps_sim=ballistic_40u.shape[1],
                eps=cfg.metrics.sampled_average_state.eps,
            ),
        ]
    )
    brownian_to_brownian_similarity = np.array(
        [
            compute_sampled_average_state_measure(
                brownian_40b,
                brownian_40b,
                num_time_windows=num_windows,
                num_steps_sim=brownian_40b.shape[1],
                eps=cfg.metrics.sampled_average_state.eps,
            ),
            compute_sampled_average_state_measure(
                brownian_30b,
                brownian_30b,
                num_time_windows=num_windows,
                num_steps_sim=brownian_30b.shape[1],
                eps=cfg.metrics.sampled_average_state.eps,
            ),
            compute_sampled_average_state_measure(
                brownian_40u,
                brownian_40u,
                num_time_windows=num_windows,
                num_steps_sim=brownian_40u.shape[1],
                eps=cfg.metrics.sampled_average_state.eps,
            ),
        ]
    )

    df = open_dataframe(
        csv_path_name,
        [
            "similarity_type",
            "metrics",
            "similarity_score_mean",
            "similarity_score_std",
        ],
    )
    df = append_row_to_dataframe(
        df,
        {
            "similarity_type": "Reynolds to Reynolds",
            "metrics": similarity_metrics,
            "similarity_score_mean": np.round(
                np.mean(reynolds_to_reynolds_similarity), 2
            ),
            "similarity_score_std": np.round(
                np.std(reynolds_to_reynolds_similarity), 2
            ),
        },
    )
    df = append_row_to_dataframe(
        df,
        {
            "similarity_type": "Reynolds to Vicsek",
            "metrics": similarity_metrics,
            "similarity_score_mean": np.round(
                np.mean(reynolds_to_vicsek_similarity), 2
            ),
            "similarity_score_std": np.round(np.std(reynolds_to_vicsek_similarity), 2),
        },
    )
    df = append_row_to_dataframe(
        df,
        {
            "similarity_type": "Reynolds to Aggregation",
            "metrics": similarity_metrics,
            "similarity_score_mean": np.round(
                np.mean(reynolds_to_aggregation_similarity), 2
            ),
            "similarity_score_std": np.round(
                np.std(reynolds_to_aggregation_similarity), 2
            ),
        },
    )
    df = append_row_to_dataframe(
        df,
        {
            "similarity_type": "Reynolds to Dispersion",
            "metrics": similarity_metrics,
            "similarity_score_mean": np.round(
                np.mean(reynolds_to_dispersion_similarity), 2
            ),
            "similarity_score_std": np.round(
                np.std(reynolds_to_dispersion_similarity), 2
            ),
        },
    )
    df = append_row_to_dataframe(
        df,
        {
            "similarity_type": "Reynolds to Ballistic",
            "metrics": similarity_metrics,
            "similarity_score_mean": np.round(
                np.mean(reynolds_to_ballistic_similarity), 2
            ),
            "similarity_score_std": np.round(
                np.std(reynolds_to_ballistic_similarity), 2
            ),
        },
    )
    df = append_row_to_dataframe(
        df,
        {
            "similarity_type": "Reynolds to Brownian",
            "metrics": similarity_metrics,
            "similarity_score_mean": np.round(
                np.mean(reynolds_to_brownian_similarity), 2
            ),
            "similarity_score_std": np.round(
                np.std(reynolds_to_brownian_similarity), 2
            ),
        },
    )
    df = append_row_to_dataframe(
        df,
        {
            "similarity_type": "Vicsek to Vicsek",
            "metrics": similarity_metrics,
            "similarity_score_mean": np.round(np.mean(vicsek_to_vicsek_similarity), 2),
            "similarity_score_std": np.round(np.std(vicsek_to_vicsek_similarity), 2),
        },
    )
    df = append_row_to_dataframe(
        df,
        {
            "similarity_type": "Vicsek to Aggregation",
            "metrics": similarity_metrics,
            "similarity_score_mean": np.round(
                np.mean(vicsek_to_aggregation_similarity), 2
            ),
            "similarity_score_std": np.round(
                np.std(vicsek_to_aggregation_similarity), 2
            ),
        },
    )
    df = append_row_to_dataframe(
        df,
        {
            "similarity_type": "Vicsek to Dispersion",
            "metrics": similarity_metrics,
            "similarity_score_mean": np.round(
                np.mean(vicsek_to_dispersion_similarity), 2
            ),
            "similarity_score_std": np.round(
                np.std(vicsek_to_dispersion_similarity), 2
            ),
        },
    )
    df = append_row_to_dataframe(
        df,
        {
            "similarity_type": "Vicsek to Ballistic",
            "metrics": similarity_metrics,
            "similarity_score_mean": np.round(
                np.mean(vicsek_to_ballistic_similarity), 2
            ),
            "similarity_score_std": np.round(np.std(vicsek_to_ballistic_similarity), 2),
        },
    )
    df = append_row_to_dataframe(
        df,
        {
            "similarity_type": "Vicsek to Brownian",
            "metrics": similarity_metrics,
            "similarity_score_mean": np.round(
                np.mean(vicsek_to_brownian_similarity), 2
            ),
            "similarity_score_std": np.round(np.std(vicsek_to_brownian_similarity), 2),
        },
    )
    df = append_row_to_dataframe(
        df,
        {
            "similarity_type": "Aggregation to Aggregation",
            "metrics": similarity_metrics,
            "similarity_score_mean": np.round(
                np.mean(aggregation_to_aggregation_similarity), 2
            ),
            "similarity_score_std": np.round(
                np.std(aggregation_to_aggregation_similarity), 2
            ),
        },
    )
    df = append_row_to_dataframe(
        df,
        {
            "similarity_type": "Aggregation to Dispersion",
            "metrics": similarity_metrics,
            "similarity_score_mean": np.round(
                np.mean(aggregation_to_dispersion_similarity), 2
            ),
            "similarity_score_std": np.round(
                np.std(aggregation_to_dispersion_similarity), 2
            ),
        },
    )
    df = append_row_to_dataframe(
        df,
        {
            "similarity_type": "Aggregation to Ballistic",
            "metrics": similarity_metrics,
            "similarity_score_mean": np.round(
                np.mean(aggregation_to_ballistic_similarity), 2
            ),
            "similarity_score_std": np.round(
                np.std(aggregation_to_ballistic_similarity), 2
            ),
        },
    )
    df = append_row_to_dataframe(
        df,
        {
            "similarity_type": "Aggregation to Brownian",
            "metrics": similarity_metrics,
            "similarity_score_mean": np.round(
                np.mean(aggregation_to_brownian_similarity), 2
            ),
            "similarity_score_std": np.round(
                np.std(aggregation_to_brownian_similarity), 2
            ),
        },
    )
    df = append_row_to_dataframe(
        df,
        {
            "similarity_type": "Dispersion to Dispersion",
            "metrics": similarity_metrics,
            "similarity_score_mean": np.round(
                np.mean(dispersion_to_dispersion_similarity), 2
            ),
            "similarity_score_std": np.round(
                np.std(dispersion_to_dispersion_similarity), 2
            ),
        },
    )
    df = append_row_to_dataframe(
        df,
        {
            "similarity_type": "Dispersion to Ballistic",
            "metrics": similarity_metrics,
            "similarity_score_mean": np.round(
                np.mean(dispersion_to_ballistic_similarity), 2
            ),
            "similarity_score_std": np.round(
                np.std(dispersion_to_ballistic_similarity), 2
            ),
        },
    )
    df = append_row_to_dataframe(
        df,
        {
            "similarity_type": "Dispersion to Brownian",
            "metrics": similarity_metrics,
            "similarity_score_mean": np.round(
                np.mean(dispersion_to_brownian_similarity), 2
            ),
            "similarity_score_std": np.round(
                np.std(dispersion_to_brownian_similarity), 2
            ),
        },
    )
    df = append_row_to_dataframe(
        df,
        {
            "similarity_type": "Ballistic to Ballistic",
            "metrics": similarity_metrics,
            "similarity_score_mean": np.round(
                np.mean(ballistic_to_ballistic_similarity), 2
            ),
            "similarity_score_std": np.round(
                np.std(ballistic_to_ballistic_similarity), 2
            ),
        },
    )
    df = append_row_to_dataframe(
        df,
        {
            "similarity_type": "Ballistic to Brownian",
            "metrics": similarity_metrics,
            "similarity_score_mean": np.round(
                np.mean(ballistic_to_brownian_similarity), 2
            ),
            "similarity_score_std": np.round(
                np.std(ballistic_to_brownian_similarity), 2
            ),
        },
    )
    df = append_row_to_dataframe(
        df,
        {
            "similarity_type": "Brownian to Brownian",
            "metrics": similarity_metrics,
            "similarity_score_mean": np.round(
                np.mean(brownian_to_brownian_similarity), 2
            ),
            "similarity_score_std": np.round(
                np.std(brownian_to_brownian_similarity), 2
            ),
        },
    )
    export_dataframe(df, csv_path_name, latex_columns=None)

    log.info("\n" + 20 * "=" + "\n")


def get_npz_paths(swarm_behaviour_feats_combo) -> list:
    """"""
    return swarm_behaviour_feats_combo.npz_paths


@hydra.main(config_path="../configs", config_name="config", version_base="1.1")
def main(cfg: DictConfig):
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    hydra_cfg["runtime"]["output_dir"]

    # Redirect stdout & stderr
    sys.stdout = LoggerWriter(log, logging.INFO)
    sys.stderr = LoggerWriter(log, logging.ERROR)

    ###########################################################################
    smoothing = False
    swarm_behaviour_feats_combo = hydra.utils.instantiate(cfg.features).get_selected()
    range_generate_and_optimize = (
        range(0, 1250)
        if swarm_behaviour_feats_combo.display_name
        in ["Inspired by Gomes et al. 2013", "Hauert et al. 2022"]
        else range(250, 1500)
    )  # Since 1st 250 steps are more noisy
    load_npz_paths = get_npz_paths(swarm_behaviour_feats_combo)
    ###########################################################################

    (
        rey_swarm_metrics_40b,
        rey_swarm_metrics_30b,
        rey_swarm_metrics_40u,
        vic_swarm_metrics_40b,
        vic_swarm_metrics_30b,
        vic_swarm_metrics_40u,
        aggreg_swarm_metrics_40b,
        aggreg_swarm_metrics_30b,
        aggreg_swarm_metrics_40u,
        disper_swarm_metrics_40b,
        disper_swarm_metrics_30b,
        disper_swarm_metrics_40u,
        balli_swarm_metrics_40b,
        balli_swarm_metrics_30b,
        balli_swarm_metrics_40u,
        brown_swarm_metrics_40b,
        brown_swarm_metrics_30b,
        brown_swarm_metrics_40u,
        rey_all_f_idxs,
    ) = process_all_behaviours_variants(
        swarm_behaviour_feats_combo, range_generate_and_optimize, load_npz_paths, cfg
    )

    ######
    # Visualizing behaviours simulations per feature
    ######
    colors = {
        "1": "#1f77b4",  # blue
        "2": "#9467bd",  # purple
        "3": "#ff7f0e",  # orange
        "4": "#2ca02c",  # green
        "5": "#d62728",  # red
        "6": "#8c564b",  # brown
    }
    # visualize_behaviour_measures(
    #     rey_all_f_idxs,
    #     rey_swarm_metrics_40b[list(rey_swarm_metrics_40b.keys())[0]].shape[1],
    #     {
    #         "1": rey_swarm_metrics_40b,
    #         "2": vic_swarm_metrics_40b,
    #         "3": aggreg_swarm_metrics_40b,
    #         "4": disper_swarm_metrics_40b,
    #         "5": balli_swarm_metrics_40b,
    #         "6": brown_swarm_metrics_40b,
    #     },
    #     swarm_behaviour_feats_combo.names,
    #     colors,
    #     file_name="40b_run_{}_behaviours_signals_scatterplots.png",
    # )
    # visualize_behaviour_measures(
    #     rey_all_f_idxs,
    #     rey_swarm_metrics_30b[list(rey_swarm_metrics_30b.keys())[0]].shape[1],
    #     {
    #         "1": rey_swarm_metrics_30b,
    #         "2": vic_swarm_metrics_30b,
    #         "3": aggreg_swarm_metrics_30b,
    #         "4": disper_swarm_metrics_30b,
    #         "5": balli_swarm_metrics_30b,
    #         "6": brown_swarm_metrics_30b,
    #     },
    #     swarm_behaviour_feats_combo.names,
    #     colors,
    #     file_name="30b_run_{}_behaviours_signals_scatterplots.png",
    # )
    # visualize_behaviour_measures(
    #     rey_all_f_idxs,
    #     rey_swarm_metrics_40u[list(rey_swarm_metrics_40u.keys())[0]].shape[1],
    #     {
    #         "1": rey_swarm_metrics_40u,
    #         "2": vic_swarm_metrics_40u,
    #         "3": aggreg_swarm_metrics_40u,
    #         "4": disper_swarm_metrics_40u,
    #         "5": balli_swarm_metrics_40u,
    #         "6": brown_swarm_metrics_40u,
    #     },
    #     swarm_behaviour_feats_combo.names,
    #     colors,
    #     file_name="40u_run_{}_behaviours_signals_scatterplots.png",
    # )

    filtered_rey_swarm_metrics_40b = process_swarm_metrics_independently(
        rey_swarm_metrics_40b,
        swarm_behaviour_feats_combo.names,
        rey_all_f_idxs,
        num_sliding_w=1,
        num_overlap=0,
        applying_smoothing=smoothing,
        smoothing_level=0.1,
    )
    filtered_rey_swarm_metrics_30b = process_swarm_metrics_independently(
        rey_swarm_metrics_30b,
        swarm_behaviour_feats_combo.names,
        rey_all_f_idxs,
        num_sliding_w=1,
        num_overlap=0,
        applying_smoothing=smoothing,
        smoothing_level=0.1,
    )
    filtered_rey_swarm_metrics_40u = process_swarm_metrics_independently(
        rey_swarm_metrics_40u,
        swarm_behaviour_feats_combo.names,
        rey_all_f_idxs,
        num_sliding_w=1,
        num_overlap=0,
        applying_smoothing=smoothing,
        smoothing_level=0.1,
    )
    filtered_vic_swarm_metrics_40b = process_swarm_metrics_independently(
        vic_swarm_metrics_40b,
        swarm_behaviour_feats_combo.names,
        rey_all_f_idxs,
        num_sliding_w=1,
        num_overlap=0,
        applying_smoothing=smoothing,
        smoothing_level=0.1,
    )
    filtered_vic_swarm_metrics_30b = process_swarm_metrics_independently(
        vic_swarm_metrics_30b,
        swarm_behaviour_feats_combo.names,
        rey_all_f_idxs,
        num_sliding_w=1,
        num_overlap=0,
        applying_smoothing=smoothing,
        smoothing_level=0.1,
    )
    filtered_vic_swarm_metrics_40u = process_swarm_metrics_independently(
        vic_swarm_metrics_40u,
        swarm_behaviour_feats_combo.names,
        rey_all_f_idxs,
        num_sliding_w=1,
        num_overlap=0,
        applying_smoothing=smoothing,
        smoothing_level=0.1,
    )
    filtered_aggreg_swarm_metrics_40b = process_swarm_metrics_independently(
        aggreg_swarm_metrics_40b,
        swarm_behaviour_feats_combo.names,
        rey_all_f_idxs,
        num_sliding_w=1,
        num_overlap=0,
        applying_smoothing=smoothing,
        smoothing_level=0.1,
    )
    filtered_aggreg_swarm_metrics_30b = process_swarm_metrics_independently(
        aggreg_swarm_metrics_30b,
        swarm_behaviour_feats_combo.names,
        rey_all_f_idxs,
        num_sliding_w=1,
        num_overlap=0,
        applying_smoothing=smoothing,
        smoothing_level=0.1,
    )
    filtered_aggreg_swarm_metrics_40u = process_swarm_metrics_independently(
        aggreg_swarm_metrics_40u,
        swarm_behaviour_feats_combo.names,
        rey_all_f_idxs,
        num_sliding_w=1,
        num_overlap=0,
        applying_smoothing=smoothing,
        smoothing_level=0.1,
    )
    filtered_disper_swarm_metrics_40b = process_swarm_metrics_independently(
        disper_swarm_metrics_40b,
        swarm_behaviour_feats_combo.names,
        rey_all_f_idxs,
        num_sliding_w=1,
        num_overlap=0,
        applying_smoothing=smoothing,
        smoothing_level=0.1,
    )
    filtered_disper_swarm_metrics_30b = process_swarm_metrics_independently(
        disper_swarm_metrics_30b,
        swarm_behaviour_feats_combo.names,
        rey_all_f_idxs,
        num_sliding_w=1,
        num_overlap=0,
        applying_smoothing=smoothing,
        smoothing_level=0.1,
    )
    filtered_disper_swarm_metrics_40u = process_swarm_metrics_independently(
        disper_swarm_metrics_40u,
        swarm_behaviour_feats_combo.names,
        rey_all_f_idxs,
        num_sliding_w=1,
        num_overlap=0,
        applying_smoothing=smoothing,
        smoothing_level=0.1,
    )
    filtered_balli_swarm_metrics_40b = process_swarm_metrics_independently(
        balli_swarm_metrics_40b,
        swarm_behaviour_feats_combo.names,
        rey_all_f_idxs,
        num_sliding_w=1,
        num_overlap=0,
        applying_smoothing=smoothing,
        smoothing_level=0.1,
    )
    filtered_balli_swarm_metrics_30b = process_swarm_metrics_independently(
        balli_swarm_metrics_30b,
        swarm_behaviour_feats_combo.names,
        rey_all_f_idxs,
        num_sliding_w=1,
        num_overlap=0,
        applying_smoothing=smoothing,
        smoothing_level=0.1,
    )
    filtered_balli_swarm_metrics_40u = process_swarm_metrics_independently(
        balli_swarm_metrics_40u,
        swarm_behaviour_feats_combo.names,
        rey_all_f_idxs,
        num_sliding_w=1,
        num_overlap=0,
        applying_smoothing=smoothing,
        smoothing_level=0.1,
    )
    filtered_brown_swarm_metrics_40b = process_swarm_metrics_independently(
        brown_swarm_metrics_40b,
        swarm_behaviour_feats_combo.names,
        rey_all_f_idxs,
        num_sliding_w=1,
        num_overlap=0,
        applying_smoothing=smoothing,
        smoothing_level=0.1,
    )
    filtered_brown_swarm_metrics_30b = process_swarm_metrics_independently(
        brown_swarm_metrics_30b,
        swarm_behaviour_feats_combo.names,
        rey_all_f_idxs,
        num_sliding_w=1,
        num_overlap=0,
        applying_smoothing=smoothing,
        smoothing_level=0.1,
    )
    filtered_brown_swarm_metrics_40u = process_swarm_metrics_independently(
        brown_swarm_metrics_40u,
        swarm_behaviour_feats_combo.names,
        rey_all_f_idxs,
        num_sliding_w=1,
        num_overlap=0,
        applying_smoothing=smoothing,
        smoothing_level=0.1,
    )

    ######
    # Compute Euclidean Distance similarity
    ######
    get_euclidean_distance_sim_behaviours(
        filtered_rey_swarm_metrics_40b,
        filtered_rey_swarm_metrics_30b,
        filtered_rey_swarm_metrics_40u,
        filtered_vic_swarm_metrics_40b,
        filtered_vic_swarm_metrics_30b,
        filtered_vic_swarm_metrics_40u,
        filtered_aggreg_swarm_metrics_40b,
        filtered_aggreg_swarm_metrics_30b,
        filtered_aggreg_swarm_metrics_40u,
        filtered_disper_swarm_metrics_40b,
        filtered_disper_swarm_metrics_30b,
        filtered_disper_swarm_metrics_40u,
        filtered_balli_swarm_metrics_40b,
        filtered_balli_swarm_metrics_30b,
        filtered_balli_swarm_metrics_40u,
        filtered_brown_swarm_metrics_40b,
        filtered_brown_swarm_metrics_30b,
        filtered_brown_swarm_metrics_40u,
        similarity_metrics=(("euclidean_distance",)),
        csv_path_name="euclidean_distance_similarities_{}.csv".format(
            swarm_behaviour_feats_combo.names
        ),
        cfg=cfg,
    )

    ######
    # Compute cosine similarity
    ######
    get_cosine_sim_behaviours(
        filtered_rey_swarm_metrics_40b,
        filtered_rey_swarm_metrics_30b,
        filtered_rey_swarm_metrics_40u,
        filtered_vic_swarm_metrics_40b,
        filtered_vic_swarm_metrics_30b,
        filtered_vic_swarm_metrics_40u,
        filtered_aggreg_swarm_metrics_40b,
        filtered_aggreg_swarm_metrics_30b,
        filtered_aggreg_swarm_metrics_40u,
        filtered_disper_swarm_metrics_40b,
        filtered_disper_swarm_metrics_30b,
        filtered_disper_swarm_metrics_40u,
        filtered_balli_swarm_metrics_40b,
        filtered_balli_swarm_metrics_30b,
        filtered_balli_swarm_metrics_40u,
        filtered_brown_swarm_metrics_40b,
        filtered_brown_swarm_metrics_30b,
        filtered_brown_swarm_metrics_40u,
        similarity_metrics=(("cosine_similarity",)),
        csv_path_name="cosine_similarities_{}.csv".format(
            swarm_behaviour_feats_combo.names
        ),
    )
    # for feat1, feat2, feat3 in combinations(behaviours_swarm_metrics_lst, 3):
    #     get_cosine_sim_behaviours(...)

    ######
    # Compute Combined State Count similarity
    ######
    get_combined_state_count_sim_behaviours(
        filtered_rey_swarm_metrics_40b,
        filtered_rey_swarm_metrics_30b,
        filtered_rey_swarm_metrics_40u,
        filtered_vic_swarm_metrics_40b,
        filtered_vic_swarm_metrics_30b,
        filtered_vic_swarm_metrics_40u,
        filtered_aggreg_swarm_metrics_40b,
        filtered_aggreg_swarm_metrics_30b,
        filtered_aggreg_swarm_metrics_40u,
        filtered_disper_swarm_metrics_40b,
        filtered_disper_swarm_metrics_30b,
        filtered_disper_swarm_metrics_40u,
        filtered_balli_swarm_metrics_40b,
        filtered_balli_swarm_metrics_30b,
        filtered_balli_swarm_metrics_40u,
        filtered_brown_swarm_metrics_40b,
        filtered_brown_swarm_metrics_30b,
        filtered_brown_swarm_metrics_40u,
        similarity_metrics=(("combined_state_count",)),
        csv_path_name="combined_state_count_similarities_{}.csv".format(
            swarm_behaviour_feats_combo.names
        ),
        cfg=cfg,
    )

    ######
    # Compute Sampled Average State similarity
    ######
    for num_windows in [1, 10, 50]:
        get_sampled_average_state_sim_behaviours(
            filtered_rey_swarm_metrics_40b,
            filtered_rey_swarm_metrics_30b,
            filtered_rey_swarm_metrics_40u,
            filtered_vic_swarm_metrics_40b,
            filtered_vic_swarm_metrics_30b,
            filtered_vic_swarm_metrics_40u,
            filtered_aggreg_swarm_metrics_40b,
            filtered_aggreg_swarm_metrics_30b,
            filtered_aggreg_swarm_metrics_40u,
            filtered_disper_swarm_metrics_40b,
            filtered_disper_swarm_metrics_30b,
            filtered_disper_swarm_metrics_40u,
            filtered_balli_swarm_metrics_40b,
            filtered_balli_swarm_metrics_30b,
            filtered_balli_swarm_metrics_40u,
            filtered_brown_swarm_metrics_40b,
            filtered_brown_swarm_metrics_30b,
            filtered_brown_swarm_metrics_40u,
            similarity_metrics=(("combined_state_count",)),
            csv_path_name="sampled_average_state_similarities_{}_w_num_windows_{}.csv".format(
                swarm_behaviour_feats_combo.names, num_windows
            ),
            num_windows=num_windows,
            cfg=cfg,
        )


if __name__ == "__main__":
    main()
