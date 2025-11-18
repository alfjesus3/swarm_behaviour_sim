# import hydra
# from omegaconf import DictConfig, OmegaConf
# import logging

import matplotlib.pyplot as plt
import numpy as np
import os
import sys

from swarm_hydra.entry_point import *
from swarm_hydra.metrics.utils_metrics import *
from swarm_hydra.metrics.spatial_metrics import *
from swarm_hydra.metrics.temporal_metrics import *
from swarm_hydra.metrics.proba_metrics import *

# # A logger for this file
# log = logging.getLogger(__name__)


def get_flocking_metrics(experi_dir: str) -> None:
    """"""
    cfg_experi_rey, hdf5_experi_rey, reynolds_subfolders = processing_folder_structure(
        experi_dir_flocking, "reynolds"
    )
    cfg_experi_vic, hdf5_experi_vic, vicsek_subfolders = processing_folder_structure(
        experi_dir_flocking, "vicsek"
    )

    ### Comparing Reynolds and Vicsek per same seed
    out_dir_metrics = experi_dir + "metrics_results/"
    os.makedirs(out_dir_metrics, mode=0o777, exist_ok=True)
    metrics_all_seeds = {
        "MSE_rey": [],
        "Area_rey": [],
        "Polarisation_rey": [],
        "MSE_vic": [],
        "Area_vic": [],
        "Polarisation_vic": [],
    }
    for rey_subf, vic_subf in zip(reynolds_subfolders, vicsek_subfolders):
        # Preprocessing the simulations' data for the metrics
        rey_pos, [rey_vels] = preprocessing_data_for_metrics(
            hdf5_experi_rey[rey_subf].get("boids_pos"),
            [hdf5_experi_rey[rey_subf].get("boids_vels")],
            {
                "walls_b": cfg_experi_rey[rey_subf].behaviours.class_m.walls_b,
                "min_val": cfg_experi_rey[rey_subf].behaviours.class_m.box_init,
                "max_val": cfg_experi_rey[rey_subf].behaviours.class_m.box_size,
            },
        )
        vic_pos, [vic_vels] = preprocessing_data_for_metrics(
            hdf5_experi_vic[vic_subf].get("boids_pos"),
            [hdf5_experi_vic[vic_subf].get("boids_vels")],
            {
                "walls_b": cfg_experi_vic[vic_subf].behaviours.class_m.walls_b,
                "min_val": cfg_experi_rey[rey_subf].behaviours.class_m.box_init,
                "max_val": cfg_experi_vic[vic_subf].behaviours.class_m.box_size,
            },
        )

        print(f"Comparing the behavior in experi_folder_name {rey_subf} and {vic_subf}")

        # MSE
        mse_boids_pos1 = []
        for swarm_config_state in rey_pos:
            mse_boids_pos1.append(
                compute_mse_swarm_configuration(
                    swarm_config_state,
                    cfg_experi_rey[rey_subf].behaviours.class_m.model.alignm_dist_thres,
                    cfg_experi_rey[rey_subf].behaviours.class_m.walls_b,
                    cfg_experi_rey[rey_subf].behaviours.class_m.box_size,
                )
            )
        mse_boids_pos1_np = np.array(mse_boids_pos1)
        mse_boids_pos2 = []
        for swarm_config_state in vic_pos:
            mse_boids_pos2.append(
                compute_mse_swarm_configuration(
                    swarm_config_state,
                    cfg_experi_vic[vic_subf].behaviours.class_m.model.alignm_dist_thres,
                    cfg_experi_vic[vic_subf].behaviours.class_m.walls_b,
                    cfg_experi_vic[vic_subf].behaviours.class_m.box_size,
                )
            )
        mse_boids_pos2_np = np.array(mse_boids_pos2)
        print(
            f"The average MSE for {rey_subf} is {np.mean(mse_boids_pos1_np)} and for {vic_subf} is {np.mean(mse_boids_pos2_np)}."
        )
        metrics_all_seeds["MSE_rey"].append(mse_boids_pos1_np)
        metrics_all_seeds["MSE_vic"].append(mse_boids_pos2_np)

        # Area of the swarm configuration
        areas_boids1, areas_boids2 = [], []
        for boids1_positions, boids2_positions in zip(rey_pos, vic_pos):
            areas_boids1.append(
                compute_convex_hull_area_swarm_configuration(boids1_positions)
            )
            areas_boids2.append(
                compute_convex_hull_area_swarm_configuration(boids2_positions)
            )
        metrics_all_seeds["Area_rey"].append(areas_boids1)
        metrics_all_seeds["Area_vic"].append(areas_boids2)

        # Polarization of the swarm
        # Area of the swarm configuration
        polarisation_boids1, polarisation_boids2 = [], []
        for boids1_vels, boids2_vels in zip(rey_vels, vic_vels):
            polarisation_boids1.append(
                compute_polarisation_swarm_configuration(boids1_vels)
            )
            polarisation_boids2.append(
                compute_polarisation_swarm_configuration(boids2_vels)
            )
        metrics_all_seeds["Polarisation_rey"].append(polarisation_boids1)
        metrics_all_seeds["Polarisation_vic"].append(polarisation_boids2)

        # ...

    # Plot the mse for the swarm configurations
    mse_all_seeds_rey = np.array(metrics_all_seeds["MSE_rey"])
    mse_all_seeds_vic = np.array(metrics_all_seeds["MSE_vic"])
    iterations = np.arange(
        0, mse_all_seeds_rey.shape[1]
    )  # X-axis labels (assuming same length for both)
    _, ax0 = plt.subplots(figsize=(12, 6))  # Two subplots stacked vertically
    ax0.plot(
        iterations,
        np.mean(mse_all_seeds_rey, axis=0),
        color="blue",
        alpha=0.7,
        label=f"{rey_subf}",
    )
    ax0.errorbar(
        iterations,
        np.mean(mse_all_seeds_rey, axis=0),
        yerr=np.std(mse_all_seeds_rey, axis=0),
        fmt=".",
        markersize="0.1",
        ecolor="red",
        capsize=0.1,
        elinewidth=0.1,
    )
    ax0.set_ylabel("MSE Flocking")
    ax0.set_title("Bar Chart of Data 1")
    ax0.plot(
        iterations,
        np.mean(mse_all_seeds_vic, axis=0),
        color="green",
        alpha=0.7,
        label=f"{vic_subf}",
    )
    ax0.errorbar(
        iterations,
        np.mean(mse_all_seeds_vic, axis=0),
        yerr=np.std(mse_all_seeds_vic, axis=0),
        fmt=".",
        markersize="0.1",
        ecolor="orange",
        capsize=0.1,
        elinewidth=0.1,
    )
    ax0.set_title("Swarm configuration avg MSE over the simulation")
    ax0.set_xlabel("Iteration Number")
    iterations_subs = np.arange(0, mse_all_seeds_rey.shape[1], step=10)
    ax0.set_xticks(iterations_subs)
    ax0.set_xticklabels(iterations_subs, rotation=45, ha="right")
    ax0.legend()
    plt.tight_layout()
    plt.savefig(
        out_dir_metrics + "/mse_all_seeds.png", bbox_inches="tight"
    )  # plt.show()

    # Plot the areas of the swarm configurations
    areas_all_seeds_rey = np.array(metrics_all_seeds["Area_rey"])
    areas_all_seeds_vic = np.array(metrics_all_seeds["Area_vic"])
    iterations = np.arange(
        0, areas_all_seeds_rey.shape[1]
    )  # X-axis labels (assuming same length for both)
    _, ax = plt.subplots(figsize=(12, 6))
    ax.plot(np.mean(areas_all_seeds_rey, axis=0), color="blue", label=f"{rey_subf}")
    ax.errorbar(
        iterations,
        np.mean(areas_all_seeds_rey, axis=0),
        np.std(areas_all_seeds_rey, axis=0),
        fmt=".",
        markersize="0.1",
        ecolor="red",
        capsize=0.1,
        elinewidth=0.1,
    )
    ax.plot(np.mean(areas_all_seeds_vic, axis=0), color="green", label=f"{vic_subf}")
    ax.errorbar(
        iterations,
        np.mean(areas_all_seeds_vic, axis=0),
        np.std(areas_all_seeds_vic, axis=0),
        fmt=".",
        markersize="0.1",
        ecolor="orange",
        capsize=0.1,
        elinewidth=0.1,
    )
    ax.set_ylabel("Value")
    ax.set_title("Swarm configuration avg Area over the simulation")
    ax.set_xlabel("Iteration Number")
    iterations_subs = np.arange(0, areas_all_seeds_rey.shape[1], step=10)
    ax.set_xticks(iterations_subs)
    ax.set_xticklabels(iterations_subs, rotation=45, ha="right")
    ax.legend()
    plt.savefig(
        out_dir_metrics + "/area_all_seeds.png", bbox_inches="tight"
    )  # plt.show()

    # Plot the polarisation of the swarm configurations
    polari_all_seeds_rey = np.array(metrics_all_seeds["Polarisation_rey"])
    polari_all_seeds_vic = np.array(metrics_all_seeds["Polarisation_vic"])
    _, axes = plt.subplots(
        2, 1, sharex=True, figsize=(12, 6)
    )  # Two subplots stacked vertically
    iterations = np.array(
        [[i for _ in range(len(rey_vels[0]))] for i in range(len(rey_vels))]
    )
    axes[0].scatter(
        iterations,
        np.mean(polari_all_seeds_rey, axis=0),
        color="blue",
        s=0.1,
        label=f"{rey_subf}",
    )
    for iters, pol_rey_m, pol_rey_std in zip(
        iterations,
        np.mean(polari_all_seeds_rey, axis=0),
        np.std(polari_all_seeds_rey, axis=0),
    ):
        axes[0].errorbar(
            iters,
            pol_rey_m,
            pol_rey_std,
            fmt=".",
            markersize="0.1",
            ecolor="red",
            capsize=0.1,
            elinewidth=0.1,
        )
    axes[1].scatter(
        iterations,
        np.mean(polari_all_seeds_vic, axis=0),
        color="red",
        s=0.1,
        label=f"{vic_subf}",
    )
    for iters, pol_vic_m, pol_vic_std in zip(
        iterations,
        np.mean(polari_all_seeds_vic, axis=0),
        np.std(polari_all_seeds_vic, axis=0),
    ):
        axes[1].errorbar(
            iters,
            pol_vic_m,
            pol_vic_std,
            fmt=".",
            markersize="0.1",
            ecolor="orange",
            capsize=0.1,
            elinewidth=0.1,
        )
    axes[0].set_ylabel("SD Flocking")
    axes[0].set_ylabel("SD Flocking")
    plt.title("Swarm avg Polarisation over the simulation")
    axes[1].set_xlabel("Iteration Number")
    iterations_sub = np.arange(0, len(polarisation_boids1), step=10)
    axes[1].set_xticks(iterations_sub)
    axes[1].set_xticklabels(iterations_sub, rotation=45, ha="right")
    axes[0].legend()
    axes[1].legend()
    plt.savefig(
        out_dir_metrics + "/polarisation_all_seeds.png", bbox_inches="tight"
    )  # plt.show()

    # Polarization to Area plot for the [num_steps/4:num_steps-1] swarm configurations
    start_idx = len(areas_boids1) // 4
    _, ax = plt.subplots(figsize=(12, 6))
    min_area = np.minimum(np.min(areas_all_seeds_rey), np.min(areas_all_seeds_vic))
    max_area = np.maximum(np.max(areas_all_seeds_rey), np.max(areas_all_seeds_vic))
    # iterations = np.arange(min_area, max_area, step=1)  # X-axis labels (assuming same length for both)
    for idx, (m_area_rey, m_area_vic, m_polari_rey, m_polari_vic) in enumerate(
        zip(
            np.mean(areas_all_seeds_rey, axis=0)[start_idx:],
            np.mean(areas_all_seeds_vic, axis=0)[start_idx:],
            np.mean(polari_all_seeds_rey, axis=0)[start_idx:],
            np.mean(polari_all_seeds_vic, axis=0)[start_idx:],
        )
    ):
        m_m_polari_rey = np.mean(m_polari_rey)
        m_m_polari_vic = np.mean(m_polari_vic)
        ax.scatter(m_area_rey, m_m_polari_rey, color="blue", label=f"{rey_subf}", s=0.1)
        ax.scatter(m_area_vic, m_m_polari_vic, color="red", label=f"{vic_subf}", s=0.1)
    ax.set_ylabel("SD Flocking")
    ax.set_title("Avg Polarisation over the swarm configuration avg Area.")
    ax.set_xlabel("Area swarm configuration")
    ax.set_xlim([min_area, max_area])
    # ax.set_xticks(iterations)
    # ax.set_xticklabels(iterations, rotation=45, ha='right')
    # ax.legend()
    # Logic to filter the unique axis labels
    lines_labels = [ax.get_legend_handles_labels()]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    # grab unique labels
    unique_labels = set(labels)
    # assign labels and legends in dict
    legend_dict = dict(zip(labels, lines))
    # query dict based on unique labels
    unique_lines = [legend_dict[x] for x in unique_labels]
    ax.legend(unique_lines, unique_labels, scatterpoints=1)
    plt.savefig(
        out_dir_metrics + "/polarisation_to_area_all_seeds.png", bbox_inches="tight"
    )  # plt.show()

    # Define and export pd.DataFrame
    csv_name = "flocking_metrics.csv"
    columns = [
        "Rey_avg_MSE_tmstp",
        "Rey_avg_Area_tmstp",
        "Rey_avg_Polarisation_tmstp",
        "Rey_std_MSE_tmstp",
        "Rey_std_Area_tmstp",
        "Rey_std_Polarisation_tmstp",
        "Vic_avg_MSE_tmstp",
        "Vic_avg_Area_tmstp",
        "Vic_avg_Polarisation_tmstp",
        "Vic_std_MSE_tmstp",
        "Vic_std_Area_tmstp",
        "Vic_std_Polarisation_tmstp",
    ]
    flocking_metrics_df = open_dataframe(csv_name, columns)
    for (
        rey_avg_mse_tmstp,
        rey_avg_area_tmstp,
        rey_avg_polarisation_tmstp,
        rey_std_mse_tmstp,
        rey_std_area_tmstp,
        rey_std_polarisation_tmstp,
        vic_avg_mse_tmstp,
        vic_avg_area_tmstp,
        vic_avg_polarisation_tmstp,
        vic_std_mse_tmstp,
        vic_std_area_tmstp,
        vic_std_polarisation_tmstp,
    ) in zip(
        np.mean(mse_all_seeds_rey, axis=0),
        np.mean(areas_all_seeds_rey, axis=0),
        np.mean(polari_all_seeds_rey, axis=0),
        np.std(mse_all_seeds_rey, axis=0),
        np.std(areas_all_seeds_rey, axis=0),
        np.std(polari_all_seeds_rey, axis=0),
        np.mean(mse_all_seeds_vic, axis=0),
        np.mean(areas_all_seeds_vic, axis=0),
        np.mean(polari_all_seeds_vic, axis=0),
        np.std(mse_all_seeds_vic, axis=0),
        np.std(areas_all_seeds_vic, axis=0),
        np.std(polari_all_seeds_vic, axis=0),
    ):
        flocking_metrics_df = append_row_to_dataframe(
            flocking_metrics_df,
            {
                "Rey_avg_MSE_tmstp": rey_avg_mse_tmstp,
                "Rey_avg_Area_tmstp": rey_avg_area_tmstp,
                "Rey_avg_Polarisation_tmstp": np.mean(rey_avg_polarisation_tmstp),
                "Rey_std_MSE_tmstp": rey_std_mse_tmstp,
                "Rey_std_Area_tmstp": rey_std_area_tmstp,
                "Rey_std_Polarisation_tmstp": np.mean(rey_std_polarisation_tmstp),
                "Vic_avg_MSE_tmstp": vic_avg_mse_tmstp,
                "Vic_avg_Area_tmstp": vic_avg_area_tmstp,
                "Vic_avg_Polarisation_tmstp": np.mean(vic_avg_polarisation_tmstp),
                "Vic_std_MSE_tmstp": vic_std_mse_tmstp,
                "Vic_std_Area_tmstp": vic_std_area_tmstp,
                "Vic_std_Polarisation_tmstp": np.mean(vic_std_polarisation_tmstp),
            },
        )
    export_dataframe(
        flocking_metrics_df, out_dir_metrics + csv_name, latex_columns=None
    )

    # Export Reynolds, Vicsek hyperparameters of experiments into a latex table
    ########################
    # TODO to be debugged ....
    ########################
    hyperparams_df = yaml_to_dataframe(
        dict(
            (k, cfg_experi[k]) for k in (reynolds_subfolders[0], vicsek_subfolders[0])
        )  # cfg_experi
    )
    latex_table = hyperparams_df.to_latex(
        index=True,
        column_format="|c|c|c|",
        caption="Hyperparameter Comparison",
        label="tab:hyperparams",
    )
    with open(
        out_dir_metrics + csv_name.replace(".csv", f"_hyperparams.tex"), "w"
    ) as f:
        f.write(latex_table)


# @hydra.main(config_path="../configs", config_name="config", version_base="1.1")
def main():  # (cfg: DictConfig):
    # hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    # hydra_cfg['runtime']['output_dir']
    # out_dir = hydra_cfg['runtime']['output_dir']

    # # log.setLevel(logging.DEBUG)
    # log.info(f"Behaviour type: {cfg.behaviour}")
    # log.info(f"Output dir: {out_dir}")

    # Redirect stdout & stderr
    sys.stdout = LoggerWriter(log, logging.INFO)
    sys.stderr = LoggerWriter(log, logging.ERROR)

    get_flocking_metrics(experi_dir_flocking)


if __name__ == "__main__":
    main()
