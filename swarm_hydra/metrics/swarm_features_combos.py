import logging
import hydra
import numpy as np
from dataclasses import dataclass
from typing import List, Union
import scipy

from swarm_hydra.metrics.utils_metrics import *
from swarm_hydra.metrics.spatial_metrics import *
from swarm_hydra.metrics.temporal_metrics import *
from swarm_hydra.metrics.proba_metrics import *

# A logger for this file
log = logging.getLogger(__name__)


@dataclass
class FeaturesCombo:
    """Class representing a feature combination configuration."""

    display_name: str
    names: List[str]
    n_agents: str
    dim: int
    npz_paths: List[str]


@dataclass
class FeaturesCombos:
    """Class containing all feature combinations and the selected one."""

    Hauert2022: FeaturesCombo
    LocalBoidsFeats: FeaturesCombo
    MadeUpPreliminary: FeaturesCombo
    Yang2023: FeaturesCombo
    Kuckling2023: FeaturesCombo
    selected_combo: str

    def get_selected(self) -> FeaturesCombo:
        """Get the currently selected feature combo."""
        return getattr(self, self.selected_combo)


def get_selected_metrics(
    selected_metrics: list,
    cfg: dict,
    boids_pos: np.ndarray,
    boids_vels: np.ndarray,
    radius_sensing: float,
    store_interm_data: bool,
) -> tuple:
    """"""
    loss = []
    if store_interm_data:
        interm_data = {}
    for sel_metric in selected_metrics:
        if sel_metric == "mse_swarm":
            boids_mse = hydra.utils.instantiate(
                cfg.metrics.mse_swarm.impl,
                swarm_state=boids_pos,
                radius=radius_sensing,
                walls_b=cfg.behaviours.class_m.walls_b,
                box_size=cfg.behaviours.class_m.box_size,
            )
            metric_value = boids_mse
            if store_interm_data:
                interm_data[sel_metric] = {"metric_value": metric_value}
        elif sel_metric == "area_swarm":
            boids_area = hydra.utils.instantiate(
                cfg.metrics.area_swarm.impl, points=boids_pos
            )
            metric_value = np.mean(boids_area)
            if store_interm_data:
                interm_data[sel_metric] = {"metric_value": metric_value}
        elif sel_metric == "polarisation_swarm":
            boids_polarisation = hydra.utils.instantiate(
                cfg.metrics.polarisation_swarm.impl, boids_vels
            )
            metric_value = np.mean(boids_polarisation)
            if store_interm_data:
                interm_data[sel_metric] = {"metric_value": metric_value}
        elif sel_metric == "average_local_density":
            avg_local_d = compute_average_local_density_measure(
                boids_pos,
                radius_sensing,
                cfg.behaviours.class_m.box_size,
                cfg.behaviours.class_m.walls_b,
            )
            metric_value = avg_local_d
            if store_interm_data:
                interm_data[sel_metric] = {"metric_value": metric_value}
        elif sel_metric == "beta_index":
            beta_idx = compute_beta_index_measure(
                boids_pos,
                cfg.behaviours.class_m.box_size,
                cfg.behaviours.class_m.walls_b,
            )
            metric_value = beta_idx
            if store_interm_data:
                interm_data[sel_metric] = {"metric_value": metric_value}
        elif sel_metric == "center_of_mass":
            center_of_mass = compute_center_of_mass_swarm_configuration(boids_pos)
            metric_value = center_of_mass
            if store_interm_data:
                interm_data[sel_metric] = {"metric_value": center_of_mass}
        elif sel_metric == "maximum_swarm_shift":
            max_swarm_shift = compute_maximum_swarm_shift_swarm_configuration(
                boids_pos,
                cfg.behaviours.class_m.walls_b,
                cfg.behaviours.class_m.box_size,
            )
            metric_value = max_swarm_shift
            if store_interm_data:
                interm_data[sel_metric] = {"metric_value": max_swarm_shift}
        elif sel_metric == "swarm_mode_index":
            swarm_mode_idx = compute_swarm_mode_index_swarm_configuration(
                boids_pos, cfg.metrics.swarm_mode_index.freq_dist_threshold
            )
            metric_value = swarm_mode_idx
            if store_interm_data:
                interm_data[sel_metric] = {"metric_value": swarm_mode_idx}
        elif sel_metric == "longest_path":
            longest_path = compute_longest_path_swarm_configuration(boids_pos)
            metric_value = longest_path
            if store_interm_data:
                interm_data[sel_metric] = {"metric_value": longest_path}
        elif sel_metric == "maximum_radius":
            maximum_radius = compute_maximum_radius_swarm_configuration(boids_pos)
            metric_value = maximum_radius
            if store_interm_data:
                interm_data[sel_metric] = {"metric_value": maximum_radius}
        elif sel_metric == "average_nearest_neighbour_distance":
            average_nearest_neighbour_distance = (
                compute_average_nearest_neighbour_distance_measure(boids_pos)
            )
            metric_value = average_nearest_neighbour_distance
            if store_interm_data:
                interm_data[sel_metric] = {
                    "metric_value": average_nearest_neighbour_distance
                }
        elif sel_metric == "collision_count":
            collision_count = compute_collision_count_measure(
                boids_pos,
                cfg.behaviours.flocking.reynolds.alignm_dist_thres,
                cfg.behaviours.class_m.walls_b,
                cfg.behaviours.class_m.box_size,
            )
            metric_value = collision_count
            if store_interm_data:
                interm_data[sel_metric] = {"metric_value": collision_count}
        elif sel_metric == "flock_density":
            flock_density = compute_flock_density_measure(boids_pos)
            metric_value = flock_density
            if store_interm_data:
                interm_data[sel_metric] = {"metric_value": flock_density}
        elif sel_metric == "grouping":
            grouping = compute_grouping_measure(
                boids_pos,
                cfg.behaviours.class_m.walls_b,
                cfg.behaviours.class_m.box_size,
            )
            metric_value = grouping
            if store_interm_data:
                interm_data[sel_metric] = {"metric_value": grouping}
        elif sel_metric == "straggler_count":
            straggler_count = compute_straggler_count_measure(
                boids_pos,
                cfg.behaviours.flocking.reynolds.alignm_dist_thres,
                cfg.behaviours.class_m.walls_b,
                cfg.behaviours.class_m.box_size,
            )
            metric_value = straggler_count
            if store_interm_data:
                interm_data[sel_metric] = {"metric_value": straggler_count}
        elif sel_metric == "order":
            order = compute_order_measure(boids_vels)
            metric_value = order
            if store_interm_data:
                interm_data[sel_metric] = {"metric_value": order}
        elif sel_metric == "subgroup_count":
            subgroup_count = compute_subgroup_count_measure(
                boids_pos,
                cfg.behaviours.flocking.reynolds.alignm_dist_thres,
                cfg.behaviours.class_m.walls_b,
                cfg.behaviours.class_m.box_size,
            )
            metric_value = subgroup_count
            if store_interm_data:
                interm_data[sel_metric] = {"metric_value": subgroup_count}
        elif sel_metric == "diffusion":
            diffusion = compute_diffusion_measure(
                boids_pos, cfg.metrics.diffusion.time_lag
            )
            metric_value = diffusion
            if store_interm_data:
                interm_data[sel_metric] = {"metric_value": diffusion}
        elif sel_metric == "neighbor_shortest_distances":
            arena_radius = np.sqrt(
                2
                * (
                    (cfg.behaviours.class_m.box_init + cfg.behaviours.class_m.box_size)
                    // 2
                )
                ** 2
            )  # since it must fit entirely the square arena currently used
            neighbor_shortest_distances = compute_neighbor_shortest_distances_measure(
                boids_pos,
                arena_radius,
                cfg.metrics.neighbor_shortest_distances.desc_sort,
                cfg.behaviours.class_m.walls_b,
                cfg.behaviours.class_m.box_size,
            )
            metric_value = neighbor_shortest_distances
            if store_interm_data:
                interm_data[sel_metric] = {"metric_value": neighbor_shortest_distances}
        else:
            raise NotImplementedError

        loss.append(metric_value)

    if store_interm_data:
        return loss, interm_data
    else:
        return loss


def get_metrics_combo(feats_combo_n: str, **kwargs) -> tuple:
    """"""
    res = None

    if feats_combo_n.n_agents == 1:
        res = get_selected_metrics(
            feats_combo_n.names,
            cfg=kwargs.pop("cfg"),
            boids_pos=kwargs.pop("boids_pos"),
            boids_vels=kwargs.pop("boids_vels"),
            radius_sensing=kwargs.pop("radius_sensing"),
            store_interm_data=kwargs.pop("store_interm_data"),
        )
    elif feats_combo_n.display_name == "Kuckling et al. 2023":
        res = np.array(
            get_selected_metrics(
                feats_combo_n.names,
                cfg=kwargs.pop("cfg"),
                boids_pos=kwargs.pop("boids_pos"),
                boids_vels=kwargs.pop("boids_vels"),
                radius_sensing=kwargs.pop("radius_sensing"),
                store_interm_data=kwargs.pop("store_interm_data"),
            )
        ).T
    elif feats_combo_n.display_name == "Inspired by Gomes et al. 2013":
        res = np.concatenate(
            (kwargs.pop("boids_pos"), kwargs.pop("boids_vels")), axis=-1
        )
    else:
        raise NotImplementedError

    return res
