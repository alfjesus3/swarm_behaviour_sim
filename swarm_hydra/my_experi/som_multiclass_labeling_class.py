import hydra
from omegaconf import DictConfig, OmegaConf
import logging

import numpy as np
import sys
import os

from minisom import MiniSom
import minisom
import matplotlib.pyplot as plt
import matplotlib
import itertools
from tqdm import tqdm
import h5py
import gc
from typing import Callable
from collections import defaultdict, Counter
from itertools import combinations
from copy import deepcopy

from swarm_hydra.entry_point import *
from swarm_hydra.metrics.utils_metrics import (
    open_dataframe,
    append_row_to_dataframe,
    export_dataframe,
)
from swarm_hydra.my_experi.multiclass_class_experi import *

import sklearn

# A logger for this file
log = logging.getLogger(__name__)


def classify(som, data, x_train, y_train, labelling_type="majority_voting"):
    """Classifies each sample in data using different labeling methods.

    Parameters:
    -----------
    som : SOM object
        Self-organizing map instance
    data : array-like
        Data samples to classify
    x_train : array-like
        Training data used to create the labeling
    y_train : array-like
        Training labels corresponding to x_train
    labelling_type : str
        Type of labeling method to use:
        - 'majority_voting': Most frequent class among samples mapped to neuron
        - 'minimum_distance': Label of closest training sample to neuron
        - 'average_distance': Label of class with smallest average distance to neuron

    Returns:
    --------
    list : Classification results for each sample in data
    """

    def _classify_majority_voting(som, data, x_train, y_train):
        """Original majority voting implementation"""
        neuron_labels = som.labels_map(x_train, y_train)
        default_class = np.sum(list(neuron_labels.values())).most_common()[0][0]

        # Get Node labels
        trained_weights = som.get_weights()
        neurons_labels = np.zeros(
            (trained_weights.shape[0], trained_weights.shape[1], 1)
        )
        for i in range(trained_weights.shape[0]):
            for j in range(trained_weights.shape[1]):
                pos = (i, j)
                if pos in neuron_labels:
                    neurons_labels[i, j] = neuron_labels[pos].most_common()[0][0]
                else:
                    neurons_labels[i, j] = default_class

        # Classify train data
        train_preds = []
        for tr_pt in x_train:
            bmu = som.winner(tr_pt)
            train_preds.append(neuron_labels[bmu].most_common()[0][0])

        # Classify test data
        test_preds = []
        for d in data:
            win_position = som.winner(d)
            if win_position in neuron_labels:
                test_preds.append(neuron_labels[win_position].most_common()[0][0])
            else:
                test_preds.append(default_class)

        return neurons_labels, train_preds, test_preds

    def _classify_minimum_distance(som, data, x_train, y_train):
        """Minimum distance method: neuron inherits label of closest training sample"""
        # Create mapping from winning neurons to their closest training sample labels
        neuron_labels = {}

        # Get default class (most common overall)
        default_class = np.sum(
            list(som.labels_map(x_train, y_train).values())
        ).most_common()[0][0]

        # For each training sample, find its winning neuron and calculate distance
        for x_sample, y_label in zip(x_train, y_train):
            win_pos = som.winner(x_sample)

            # Calculate distance using SOM's own distance function
            distance = som.activate(x_sample)[win_pos]

            # Keep track of closest sample for each neuron
            if win_pos not in neuron_labels or distance < neuron_labels[win_pos][1]:
                neuron_labels[win_pos] = (y_label, distance)

        # Extract just the labels (remove distances)
        neuron_to_label = {pos: label for pos, (label, _) in neuron_labels.items()}

        # Get Node labels
        trained_weights = som.get_weights()
        neurons_labels = np.zeros(
            (trained_weights.shape[0], trained_weights.shape[1], 1)
        )
        for i in range(trained_weights.shape[0]):
            for j in range(trained_weights.shape[1]):
                pos = (i, j)
                if pos in neuron_to_label:
                    neurons_labels[i, j] = neuron_to_label[pos]
                else:
                    neurons_labels[i, j] = default_class

        # Classify train data
        train_preds = []
        for tr_pt in x_train:
            win_position = som.winner(tr_pt)
            if win_position in neuron_to_label:
                train_preds.append(neuron_to_label[win_position])
            else:
                train_preds.append(default_class)

        # Classify test data
        test_preds = []
        for d in data:
            win_position = som.winner(d)
            if win_position in neuron_to_label:
                test_preds.append(neuron_to_label[win_position])
            else:
                test_preds.append(default_class)
        return neurons_labels, train_preds, test_preds

    def _classify_average_distance(som, data, x_train, y_train):
        """Average distance method: neuron gets label of class with smallest average distance"""
        # Group training samples by their winning neurons
        neuron_samples = defaultdict(lambda: defaultdict(list))

        # Get default class (most common overall)
        default_class = np.sum(
            list(som.labels_map(x_train, y_train).values())
        ).most_common()[0][0]

        for x_sample, y_label in zip(x_train, y_train):
            win_pos = som.winner(x_sample)
            neuron_samples[win_pos][y_label].append(x_sample)

        # Calculate average distances and assign labels
        neuron_to_label = {}

        for win_pos, class_samples in neuron_samples.items():
            min_avg_distance = float("inf")
            best_class = None

            # Calculate average distance for each class mapped to this neuron
            for class_label, samples in class_samples.items():
                distances = [som.activate(sample)[win_pos] for sample in samples]
                avg_distance = np.mean(distances)

                if avg_distance < min_avg_distance:
                    min_avg_distance = avg_distance
                    best_class = class_label

            neuron_to_label[win_pos] = best_class

        # Get Node labels
        trained_weights = som.get_weights()
        neurons_labels = np.zeros(
            (trained_weights.shape[0], trained_weights.shape[1], 1)
        )
        for i in range(trained_weights.shape[0]):
            for j in range(trained_weights.shape[1]):
                pos = (i, j)
                if pos in neuron_to_label:
                    neurons_labels[i, j] = neuron_to_label[pos]
                else:
                    neurons_labels[i, j] = default_class

        # Classify train data
        train_preds = []
        for tr_pt in x_train:
            win_position = som.winner(tr_pt)
            if win_position in neuron_to_label:
                train_preds.append(neuron_to_label[win_position])
            else:
                train_preds.append(default_class)

        # Classify test data
        test_preds = []
        for d in data:
            win_position = som.winner(d)
            if win_position in neuron_to_label:
                test_preds.append(neuron_to_label[win_position])
            else:
                test_preds.append(default_class)

        return neurons_labels, train_preds, test_preds

    if labelling_type == "majority_voting":
        return _classify_majority_voting(som, data, x_train, y_train)
    elif labelling_type == "minimum_distance":
        return _classify_minimum_distance(som, data, x_train, y_train)
    elif labelling_type == "average_distance":
        return _classify_average_distance(som, data, x_train, y_train)
    else:
        raise ValueError(
            f"Unknown labelling_type: {labelling_type}. "
            "Must be 'majority_voting', 'minimum_distance', or 'average_distance'"
        )


def get_som_stats(som, tr_data, tr_num_target, te_data, te_num_target) -> tuple:
    """"""
    log.info("With method='majority_voting'")
    _, tr_preds_mv, te_preds_mv = classify(
        som, te_data, tr_data, tr_num_target, "majority_voting"
    )
    try:
        tr_acc_mv = np.round(
            sklearn.metrics.accuracy_score(tr_num_target, tr_preds_mv), 3
        )
    except Exception:
        tr_acc_mv = np.nan
    try:
        tr_jac_mv = np.round(
            sklearn.metrics.jaccard_score(tr_num_target, tr_preds_mv, average="macro"),
            3,
        )
    except Exception:
        tr_jac_mv = np.nan
    try:
        tr_pre_mv = np.round(
            sklearn.metrics.precision_score(
                tr_num_target,
                tr_preds_mv,
                labels=[i for i in range(tr_num_target.shape[0])],
                average="macro",
            ),
            3,
        )
    except Exception:
        tr_pre_mv = np.nan
    try:
        tr_rec_mv = np.round(
            sklearn.metrics.recall_score(
                tr_num_target,
                tr_preds_mv,
                labels=[i for i in range(tr_num_target.shape[0])],
                average="macro",
            ),
            3,
        )
    except Exception:
        tr_rec_mv = np.nan
    log.info(
        ">>> train acc {} jac {} prec {} rec {}".format(
            tr_acc_mv,
            tr_jac_mv,
            tr_pre_mv,
            tr_rec_mv,
        )
    )
    try:
        te_acc_mv = np.round(
            sklearn.metrics.accuracy_score(te_num_target, te_preds_mv), 3
        )
    except Exception:
        te_acc_mv = np.nan
    try:
        te_jac_mv = np.round(
            sklearn.metrics.jaccard_score(te_num_target, te_preds_mv, average="macro"),
            3,
        )
    except Exception:
        te_jac_mv = np.nan
    try:
        te_pre_mv = np.round(
            sklearn.metrics.precision_score(
                te_num_target,
                te_preds_mv,
                labels=[i for i in range(te_num_target.shape[0])],
                average="macro",
            ),
            3,
        )
    except Exception:
        te_pre_mv = np.nan
    try:
        te_rec_mv = np.round(
            sklearn.metrics.recall_score(
                te_num_target,
                te_preds_mv,
                labels=[i for i in range(te_num_target.shape[0])],
                average="macro",
            ),
            3,
        )
    except Exception:
        te_rec_mv = np.nan
    log.info(
        ">>> test acc {} jac {} prec {} rec {}".format(
            te_acc_mv,
            te_jac_mv,
            te_pre_mv,
            te_rec_mv,
        )
    )
    log.info("With method='minimum_distance'")
    _, tr_preds_md, te_preds_md = classify(
        som, te_data, tr_data, tr_num_target, "minimum_distance"
    )
    try:
        tr_acc_md = np.round(
            sklearn.metrics.accuracy_score(tr_num_target, tr_preds_md), 3
        )
    except Exception:
        tr_acc_md = np.nan
    try:
        tr_jac_md = np.round(
            sklearn.metrics.jaccard_score(tr_num_target, tr_preds_md, average="macro"),
            3,
        )
    except Exception:
        tr_jac_md = np.nan
    try:
        tr_pre_md = np.round(
            sklearn.metrics.precision_score(
                tr_num_target,
                tr_preds_md,
                labels=[i for i in range(tr_num_target.shape[0])],
                average="macro",
            ),
            3,
        )
    except Exception:
        tr_pre_md = np.nan
    try:
        tr_rec_md = np.round(
            sklearn.metrics.recall_score(
                tr_num_target,
                tr_preds_md,
                labels=[i for i in range(tr_num_target.shape[0])],
                average="macro",
            ),
            3,
        )
    except Exception:
        tr_rec_md = np.nan
    log.info(
        ">>> train acc {} jac {} prec {} rec {}".format(
            tr_acc_md,
            tr_jac_md,
            tr_pre_md,
            tr_rec_md,
        )
    )
    try:
        te_acc_md = np.round(
            sklearn.metrics.accuracy_score(te_num_target, te_preds_md), 3
        )
    except Exception:
        te_acc_md = np.nan
    try:
        te_jac_md = np.round(
            sklearn.metrics.jaccard_score(te_num_target, te_preds_md, average="macro"),
            3,
        )
    except Exception:
        te_jac_md = np.nan
    try:
        te_pre_md = np.round(
            sklearn.metrics.precision_score(
                te_num_target,
                te_preds_md,
                labels=[i for i in range(te_num_target.shape[0])],
                average="macro",
            ),
            3,
        )
    except Exception:
        te_pre_md = np.nan
    try:
        te_rec_md = np.round(
            sklearn.metrics.recall_score(
                te_num_target,
                te_preds_md,
                labels=[i for i in range(te_num_target.shape[0])],
                average="macro",
            ),
            3,
        )
    except Exception:
        te_rec_md = np.nan
    log.info(
        ">>> test acc {} jac {} prec {} rec {}".format(
            te_acc_md,
            te_jac_md,
            te_pre_md,
            te_rec_md,
        )
    )
    log.info("With method='average_distance'")
    _, tr_preds_ad, te_preds_ad = classify(
        som, te_data, tr_data, tr_num_target, "average_distance"
    )
    try:
        tr_acc_ad = np.round(
            sklearn.metrics.accuracy_score(tr_num_target, tr_preds_ad), 3
        )
    except Exception:
        tr_acc_ad = np.nan
    try:
        tr_jac_ad = np.round(
            sklearn.metrics.jaccard_score(tr_num_target, tr_preds_ad, average="macro"),
            3,
        )
    except Exception:
        tr_jac_ad = np.nan
    try:
        tr_pre_ad = np.round(
            sklearn.metrics.precision_score(
                tr_num_target,
                tr_preds_ad,
                labels=[i for i in range(tr_num_target.shape[0])],
                average="macro",
            ),
            3,
        )
    except Exception:
        tr_pre_ad = np.nan
    try:
        tr_rec_ad = np.round(
            sklearn.metrics.recall_score(
                tr_num_target,
                tr_preds_ad,
                labels=[i for i in range(tr_num_target.shape[0])],
                average="macro",
            ),
            3,
        )
    except Exception:
        tr_rec_ad = np.nan
    log.info(
        ">>> train acc {} jac {} prec {} rec {}".format(
            tr_acc_ad,
            tr_jac_ad,
            tr_pre_ad,
            tr_rec_ad,
        )
    )
    try:
        te_acc_ad = np.round(
            sklearn.metrics.accuracy_score(te_num_target, te_preds_ad), 3
        )
    except Exception:
        te_acc_ad = np.nan
    try:
        te_jac_ad = np.round(
            sklearn.metrics.jaccard_score(te_num_target, te_preds_ad, average="macro"),
            3,
        )
    except Exception:
        te_jac_ad = np.nan
    try:
        te_pre_ad = np.round(
            sklearn.metrics.precision_score(
                te_num_target,
                te_preds_ad,
                labels=[i for i in range(te_num_target.shape[0])],
                average="macro",
            ),
            3,
        )
    except Exception:
        te_pre_ad = np.nan
    try:
        te_rec_ad = np.round(
            sklearn.metrics.recall_score(
                te_num_target,
                te_preds_ad,
                labels=[i for i in range(te_num_target.shape[0])],
                average="macro",
            ),
            3,
        )
    except Exception:
        te_rec_ad = np.nan
    log.info(
        ">>> test acc {} jac {} prec {} rec {}".format(
            te_acc_ad,
            te_jac_ad,
            te_pre_ad,
            te_rec_ad,
        )
    )

    return (
        tr_acc_mv,
        te_acc_mv,
        tr_acc_md,
        te_acc_md,
        tr_acc_ad,
        te_acc_ad,
    )


def my_som_train2(
    som,
    tr_data,
    tr_num_target,
    te_data,
    te_num_target,
    num_iteration,
    use_epochs=False,
    patience=6,  # Number of checks without improvement before stopping
    min_improvement=0.001,  # Minimum improvement threshold
) -> tuple:
    """
    Adapted from MiniSom.train() logic and end of https://github.com/JustGlowing/minisom/blob/master/examples/BasicUsage.ipynb
    Added early stopping based on test accuracy stalling during training.
    """
    # Classify at the beginning
    log.info(">>> At t=0 vv")
    (
        tr_acc_mv_start,
        te_acc_mv_start,
        tr_acc_md_start,
        te_acc_md_start,
        tr_acc_ad_start,
        te_acc_ad_start,
    ) = get_som_stats(som, tr_data, tr_num_target, te_data, te_num_target)

    iterations = np.arange(num_iteration) % len(tr_data)

    if use_epochs:

        def get_decay_rate(iteration_index, data_len):
            return int(iteration_index / data_len)

    else:

        def get_decay_rate(iteration_index, data_len):
            return int(iteration_index)

    fixed_points = {}

    # Early stopping setup
    check_points = np.linspace(0, len(iterations) - 1, 10, dtype=int)
    best_te_acc = te_acc_mv_start
    no_improvement_count = 0
    te_acc_history = [te_acc_mv_start]

    temp_prev_w = None
    for t, iteration in enumerate(iterations):
        if (t == 0) or (t + 1 == iterations.shape[0]):
            if temp_prev_w is not None:
                log.info(
                    "Weights change mean {} std {}".format(
                        np.mean(
                            np.subtract(temp_prev_w, som.get_weights()), axis=(0, 1)
                        ),
                        np.std(
                            np.subtract(temp_prev_w, som.get_weights()), axis=(0, 1)
                        ),
                    )
                )
            temp_prev_w = deepcopy(som.get_weights())

        decay_rate = get_decay_rate(t, len(tr_data))
        som.update(
            tr_data[iteration],
            fixed_points.get(iteration, som.winner(tr_data[iteration])),
            decay_rate,
            num_iteration,
        )

        # Check for overfitting at designated checkpoints
        if t in check_points and t > 0:  # Skip the first checkpoint (t=0)
            tr_acc_current, te_acc_current, _, _, _, _ = get_som_stats(
                som, tr_data, tr_num_target, te_data, te_num_target
            )

            te_acc_history.append(te_acc_current)

            log.info(
                f">>> Checkpoint at t={t+1}: Train Acc: {tr_acc_current:.4f}, Test Acc: {te_acc_current:.4f}"
            )

            # Check if test accuracy improved
            if te_acc_current > best_te_acc + min_improvement:
                best_te_acc = te_acc_current
                no_improvement_count = 0
                log.info(f"Test accuracy improved to {te_acc_current:.4f}")
            else:
                no_improvement_count += 1
                log.info(
                    f"Test accuracy stalled. No improvement count: {no_improvement_count}/{patience}"
                )

            # Early stopping condition
            if no_improvement_count >= patience:
                log.info(
                    f"Early stopping triggered at iteration {t+1}. Test accuracy has not improved for {patience} consecutive checks."
                )
                log.info(f"Best test accuracy: {best_te_acc:.4f}")
                log.info(
                    f"Test accuracy history: {[f'{acc:.4f}' for acc in te_acc_history]}"
                )
                break

    # Classify after the training (or early stop)
    log.info(f">>> At t={t + 1} vv")
    (
        tr_acc_mv_end,
        te_acc_mv_end,
        tr_acc_md_end,
        te_acc_md_end,
        tr_acc_ad_end,
        te_acc_ad_end,
    ) = get_som_stats(som, tr_data, tr_num_target, te_data, te_num_target)

    return (
        som,
        (tr_acc_mv_start, tr_acc_mv_end),
        (te_acc_mv_start, te_acc_mv_end),
        (tr_acc_md_start, tr_acc_md_end),
        (te_acc_md_start, te_acc_md_end),
        (tr_acc_ad_start, tr_acc_ad_end),
        (te_acc_ad_start, te_acc_ad_end),
    )


def brute_force_som_hp_search(
    num_nodes_per_dim: list,
    learning_rate: list,
    sigma: list,
    weight_init: list,
    neighborhood_function: list,
    topology: list,
    activation_distance: list,
    decay_function: list,
    sigma_decay_function: list,
    tr_data: np.ndarray,
    tr_target_num: np.ndarray,
    te_data: np.ndarray,
    te_target_num: np.ndarray,
    num_iters: int,
    num_sliding_w: int,
    colors: dict,
    swarm_metrics: list,
    label_names: list,
    random_seed: int,
    file_path: str,
) -> tuple:
    """"""
    # Generate all combinations
    all_combinations = list(
        itertools.product(
            num_nodes_per_dim,
            learning_rate,
            sigma,
            weight_init,
            neighborhood_function,
            topology,
            activation_distance,
            decay_function,
            sigma_decay_function,
        )
    )
    # Convert each combination to a list (optional, since itertools.product gives tuples)
    all_combinations_as_lists = [list(comb) for comb in all_combinations]

    tr_vis_rand_idxs = np.random.choice(
        tr_data.shape[0],
        size=tr_data.shape[0] // 20,
        replace=False,
    )
    te_vis_rand_idxs = np.random.choice(
        te_data.shape[0],
        size=tr_data.shape[0] // 20,
        replace=False,
    )

    csv_path_name = "#_brute_force_som_hp_search_combos_res.csv"
    combos_res_df = open_dataframe(
        csv_path_name,
        [
            "combo",
            "<_quantization_&_topographic_tr",
            "<_quantization_&_topographic_te",
            ">_accuracy_tr",
            ">_accuracy_te",
            "<_both_tr",
            "<_both_te",
            ">_maj_vot_tr",
            ">_maj_vot_te",
            ">_min_dist_tr",
            ">_min_dist_te",
            ">_avg_dist_tr",
            ">_avg_dist_te",
        ],
    )

    init_combos = {}
    tr_losses, te_losses = [], []
    highest_tr_te_acc, highest_tr_te_acc_combo = np.inf, None
    with tqdm(total=len(all_combinations_as_lists), desc="HP Optimization") as pbar:
        for idx_c, (hp_combo) in enumerate(all_combinations_as_lists):
            n_n_c, lr_c, sigma_c, init_c, nei_c, top_c, act_c, dec_c, sig_dec_c = (
                hp_combo
            )
            som = MiniSom(
                input_len=tr_data.shape[1],
                x=n_n_c,
                y=n_n_c,
                random_seed=random_seed,
                sigma=sigma_c * n_n_c,
                learning_rate=lr_c,
                neighborhood_function=nei_c,
                topology=top_c,
                activation_distance=act_c,
                decay_function=dec_c,
                sigma_decay_function=sig_dec_c,
            )
            som._weights = np.clip(som._weights, 0, 1)
            if init_combos.get(f"{n_n_c}-{init_c}-{top_c}") is not None:
                som._weights = deepcopy(init_combos.get(f"{n_n_c}-{init_c}-{top_c}"))
            if init_c == "random_s":
                if init_combos.get(f"{n_n_c}-{init_c}-{top_c}") is not None:
                    som._weights = deepcopy(
                        init_combos.get(f"{n_n_c}-{init_c}-{top_c}")
                    )
                else:
                    som.random_weights_init(tr_data)
            elif init_c == "pca":
                if init_combos.get(f"{n_n_c}-{init_c}-{top_c}") is not None:
                    som._weights = deepcopy(
                        init_combos.get(f"{n_n_c}-{init_c}-{top_c}")
                    )
                else:
                    som.pca_weights_init(tr_data)
            elif init_c != "random_w":
                raise NotImplementedError
            if init_combos.get(f"{n_n_c}-{init_c}-{top_c}") is None:
                init_combos[f"{n_n_c}-{init_c}-{top_c}"] = deepcopy(som._weights)

            (
                som,
                tr_accs_mv,
                te_accs_mv,
                tr_accs_md,
                te_accs_md,
                tr_accs_ad,
                te_accs_ad,
            ) = my_som_train2(
                som,
                tr_data,
                tr_target_num,
                te_data,
                te_target_num,
                num_iters,
                use_epochs=False,
            )  # som.train(tr_data, num_iters, random_order=False, verbose=False, use_epochs=False)

            log.info(
                "(idx_c= {}; {}) tr_quant {} tr_topo {} tr_dist {} te_quant {} te_topo {} te_dist {}".format(
                    idx_c,
                    hp_combo,
                    som.quantization_error(tr_data),
                    som.topographic_error(tr_data),
                    som.distortion_measure(tr_data),
                    som.quantization_error(te_data),
                    som.topographic_error(te_data),
                    som.distortion_measure(te_data),
                )
            )

            tr_som_errors = np.abs(som.quantization_error(tr_data)) + np.abs(
                som.topographic_error(tr_data)
            )
            te_som_errors = np.abs(som.quantization_error(te_data)) + np.abs(
                som.topographic_error(te_data)
            )
            if (
                tr_som_errors == float("inf")
                or np.isnan(tr_som_errors)
                or te_som_errors == float("inf")
                or np.isnan(te_som_errors)
            ):
                # Delete weights, SOM and other objects to free memory
                del som
                gc.collect()
                pbar.update(1)
                continue  # Skip the rest due to infinite values

            som_nodes_labels, _, _ = classify(
                som, te_data, tr_data, tr_target_num, "majority_voting"
            )
            tr_losses.append((1 - tr_accs_mv[1]) + tr_som_errors)
            te_losses.append((1 - te_accs_mv[1]) + te_som_errors)

            combos_res_df = append_row_to_dataframe(
                combos_res_df,
                {
                    "combo": hp_combo,
                    "<_quantization_&_topographic_tr": tr_som_errors,
                    "<_quantization_&_topographic_te": te_som_errors,
                    ">_accuracy_tr": tr_accs_mv[1],
                    ">_accuracy_te": te_accs_mv[1],
                    "<_both_tr": (1 - tr_accs_mv[1]) + tr_som_errors,
                    "<_both_te": (1 - te_accs_mv[1]) + te_som_errors,
                    ">_maj_vot_tr": tr_accs_mv[1],
                    ">_maj_vot_te": te_accs_mv[1],
                    ">_min_dist_tr": tr_accs_md[1],
                    ">_min_dist_te": te_accs_md[1],
                    ">_avg_dist_tr": tr_accs_ad[1],
                    ">_avg_dist_te": te_accs_ad[1],
                },
            )

            if (tr_losses[-1] + te_losses[-1]) < highest_tr_te_acc:
                highest_tr_te_acc = tr_losses[-1] + te_losses[-1]
                highest_tr_te_acc_combo = hp_combo
                log.info(
                    "(idx_c= {}) The new lowest error combo is {} with a score of {}.".format(
                        idx_c, highest_tr_te_acc_combo, highest_tr_te_acc
                    )
                )

            # Visualizing after training
            som_trained_w = som.get_weights()

            # visualize_samples_per_neuron(
            #     som_labels_map=som.labels_map(
            #         tr_data[tr_vis_rand_idxs],
            #         tr_target_num[tr_vis_rand_idxs].astype(str),
            #     ),
            #     label_names=label_names,
            #     gt_labels=som_nodes_labels,
            #     colors=colors,
            #     file_name=f"som_multiclass_labeling_class_brute_force_som_hp_search_visualize_samples_per_neuron_tr_{idx_c}.png",
            # )
            # visualize_samples_per_neuron(
            #     som_labels_map=som.labels_map(
            #         te_data[te_vis_rand_idxs],
            #         te_target_num[te_vis_rand_idxs].astype(str),
            #     ),
            #     label_names=label_names,
            #     gt_labels=som_nodes_labels,
            #     colors=colors,
            #     file_name=f"som_multiclass_labeling_class_brute_force_som_hp_search_visualize_samples_per_neuron_te_{idx_c}.png",
            # )

            # som_x_r = (
            #     np.min(som.get_euclidean_coordinates()[0]),
            #     np.max(som.get_euclidean_coordinates()[0]),
            # )
            # som_y_r = (
            #     np.min(som.get_euclidean_coordinates()[1]),
            #     np.max(som.get_euclidean_coordinates()[1]),
            # )
            # if top_c == "rectangular":
            #     visualize_rect_u_matrix(
            #         som.distance_map(),
            #         file_path + f"visualize_rect_u_matrix_{idx_c}.png",
            #         som_x_r,
            #         som_y_r,
            #     )
            # elif top_c == "hexagonal":
            #     visualize_hexag_u_matrix(
            #         som.get_euclidean_coordinates(),
            #         som.distance_map(),
            #         som_trained_w,
            #         file_path + f"visualize_hexag_u_matrix_{idx_c}.png",
            #         som_x_r,
            #         som_y_r,
            #     )
            # else:
            #     raise NotImplementedError

            # if num_sliding_w == 1:
            #     visualize_feature_influence(
            #         som_trained_w,
            #         swarm_metrics,
            #         file_name=file_path + f"visualize_feat_influence_{idx_c}.png",
            #     )
            # else:
            #     visualize_time_series_feature_influence(
            #         som_trained_w,
            #         swarm_metrics,
            #         file_name=file_path
            #         + f"visualize_time_series_feat_influence_{idx_c}.png",
            #         x_range=som_x_r,
            #         y_range=som_y_r,
            #     )

            # Delete weights, SOM and other objects to free memory
            del som_trained_w
            del som
            gc.collect()
            pbar.update(1)

    export_dataframe(combos_res_df, csv_path_name, latex_columns=None)

    return (
        highest_tr_te_acc_combo,
        highest_tr_te_acc,
        tr_losses,
        te_losses,
        init_combos,
    )


@hydra.main(config_path="../configs", config_name="config", version_base="1.1")
def main(cfg: DictConfig):
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    hydra_cfg["runtime"]["output_dir"]
    out_dir = hydra_cfg["runtime"]["output_dir"]

    # Redirect stdout & stderr
    sys.stdout = LoggerWriter(log, logging.INFO)
    sys.stderr = LoggerWriter(log, logging.ERROR)

    reset_seeds(cfg.behaviours.class_m.seed)

    swarm_behaviour_feats_combo = hydra.utils.instantiate(cfg.features).get_selected()
    range_generate_and_optimize = (
        range(0, 1250)
        if swarm_behaviour_feats_combo.display_name
        in ["Inspired by Gomes et al. 2013", "Hauert et al. 2022"]
        else range(250, 1500)
    )  # Since 1st 250 steps are more noisy

    num_iters_scalar = 1  # for `num_iters = int(tr_data.shape[0] * num_iters_scalar)`
    num_sliding_w = 5  # (steps // sim_freq)
    use_all_data = True

    (
        tr_data,
        tr_target,
        tr_target_num,
        tr_idxs_sub,
        te_data,
        te_target,
        te_target_num,
        te_idxs_sub,
        label_names,
        num_iters,
        num_behaviours,
        colors,
        te_f_rand_idxs_shape0,
    ) = data_processing_six_behaviours(
        range_generate_and_optimize,
        swarm_behaviour_feats_combo,
        curr_cfg=cfg,
        num_iters_scalar=num_iters_scalar,
        num_sliding_w=num_sliding_w,
        smoothing=False,
        supervised_data=False,
    )

    log.info(
        "tr_data.shape {} tr_target_num.shape {} te_data.shape {} te_target_num.shape {}".format(
            tr_data.shape,
            tr_target_num.shape,
            te_data.shape,
            te_target_num.shape,
        )
    )

    # ######
    # # SOM Hyperparameter ablation study
    # ######
    # som_size = np.round(np.sqrt(5 * np.sqrt(tr_data.shape[0]))).astype(np.int32)
    # num_nodes_per_dim = [8, 16, 32, 64]  # , 128]
    # learning_rate = [0.1]
    # sigma = [1 / 4]
    # weight_init = ["random_s"]
    # neighborhood_function = ["gaussian"]  # , "mexican_hat", "bubble"]  # , "triangle"]
    # topology = ["rectangular"]  # , "hexagonal"]
    # activation_distance = ["euclidean"]  # , "cosine", "manhattan"]  # , "chebyshev"]
    # decay_function = [
    #     "inverse_decay_to_zero",
    #     "linear_decay_to_zero",
    #     "asymptotic_decay",
    # ]  # ["linear_decay_to_zero","inverse_decay_to_zero",]  #
    # sigma_decay_function = [
    #     "inverse_decay_to_one",
    #     "linear_decay_to_one",
    #     "asymptotic_decay",
    # ]  # ["linear_decay_to_one","inverse_decay_to_one",]  #
    # (
    #     lowest_te_quant_error_combo,
    #     lowest_te_quant_error,
    #     tr_quant_error,
    #     te_quant_error,
    #     init_combos,
    # ) = brute_force_som_hp_search(
    #     num_nodes_per_dim,
    #     learning_rate,
    #     sigma,
    #     weight_init,
    #     neighborhood_function,
    #     topology,
    #     activation_distance,
    #     decay_function,
    #     sigma_decay_function,
    #     tr_data if use_all_data else tr_data[tr_idxs_sub],
    #     tr_target_num if use_all_data else tr_target_num[tr_idxs_sub],
    #     te_data if use_all_data else te_data[te_idxs_sub],
    #     te_target_num if use_all_data else te_target_num[te_idxs_sub],
    #     num_iters,
    #     num_sliding_w,
    #     colors,
    #     behaviours_swarm_metrics_lst,
    #     label_names,
    #     random_seed=cfg.behaviours.class_m.seed,
    #     file_path=out_dir + "/",
    # )

    # # Visualizing ablation study results
    # fig2 = plt.figure(figsize=(8, 8))
    # plt.plot(tr_quant_error, label="Train quantizations errors", c="b")
    # plt.plot(te_quant_error, label="Test quantizations errors", c="r")
    # # plt.vlines(
    # #     all_combinations_as_lists.index(lowest_te_quant_error_combo),
    # #     np.min(np.array(te_quant_error)),
    # #     np.max(np.array(te_quant_error)),
    # #     linestyles="dotted",
    # #     colors="k",
    # # )
    # plt.legend()
    # plt.title("SOM test quantization error")
    # plt.tight_layout()
    # fig2.savefig(
    #     "#_brute_force_som_hp_search_all_hp_quantization_error.png",
    #     format="png",
    #     bbox_inches="tight",
    #     pad_inches=0,
    # )
    # # plt.show()
    # plt.close()

    ## Best from `outputs/...` for num_sliding_w==10
    log.info(
        "n_r and n_c {}".format(
            np.round(np.sqrt(5 * np.sqrt(tr_data.shape[0]))).astype(np.int32)
        )
    )
    lowest_te_quant_error_combo = [
        np.round(np.sqrt(5 * np.sqrt(tr_data.shape[0]))).astype(np.int32),  # 64,
        0.1,
        0.25,
        "random_s",
        "gaussian",
        "rectangular",
        "euclidean",
        "inverse_decay_to_zero",
        "inverse_decay_to_one",
    ]
    lowest_te_quant_error = np.inf
    init_combos = {}

    ######
    # Re-run the training and testing with `lowest_te_quant_error_combo`
    ######
    n_n_c, lr_c, sigma_c, init_c, nei_c, top_c, act_c, dec_c, sig_dec_c = (
        lowest_te_quant_error_combo
    )
    sel_hp_som = MiniSom(
        input_len=tr_data.shape[1],
        x=n_n_c,
        y=n_n_c,
        random_seed=cfg.behaviours.class_m.seed,
        sigma=sigma_c * n_n_c,
        learning_rate=lr_c,
        neighborhood_function=nei_c,
        topology=top_c,
        activation_distance=act_c,
        decay_function=dec_c,
        sigma_decay_function=sig_dec_c,
    )
    sel_hp_som._weights = np.clip(sel_hp_som._weights, 0, 1)
    if init_combos.get(f"{n_n_c}-{init_c}-{top_c}") is not None:
        sel_hp_som._weights = deepcopy(init_combos.get(f"{n_n_c}-{init_c}-{top_c}"))
    if init_c == "random_s":
        if init_combos.get(f"{n_n_c}-{init_c}-{top_c}") is not None:
            sel_hp_som._weights = deepcopy(init_combos.get(f"{n_n_c}-{init_c}-{top_c}"))
        else:
            sel_hp_som.random_weights_init(
                tr_data if use_all_data else tr_data[tr_idxs_sub]
            )
    elif init_c == "pca":
        if init_combos.get(f"{n_n_c}-{init_c}-{top_c}") is not None:
            sel_hp_som._weights = deepcopy(init_combos.get(f"{n_n_c}-{init_c}-{top_c}"))
        else:
            sel_hp_som.pca_weights_init(
                tr_data if use_all_data else tr_data[tr_idxs_sub]
            )
    elif init_c != "random_w":
        raise NotImplementedError
    if init_combos.get(f"{n_n_c}-{init_c}-{top_c}") is None:
        init_combos[f"{n_n_c}-{init_c}-{top_c}"] = deepcopy(sel_hp_som._weights)

    sel_hp_som, _, _, _, _, _, _ = my_som_train2(
        sel_hp_som,
        tr_data if use_all_data else tr_data[tr_idxs_sub],
        tr_target_num if use_all_data else tr_target_num[tr_idxs_sub],
        te_data if use_all_data else te_data[te_idxs_sub],
        te_target_num if use_all_data else te_target_num[te_idxs_sub],
        num_iters,
        use_epochs=False,
    )  # sel_hp_som.train(tr_data, num_iters, random_order=False, verbose=False, use_epochs=False)  # sel_hp_som, q_errors, t_errors, d_errors = my_som_train(sel_hp_som, tr_data, num_iters)  #

    log.info(
        "({}) tr_quant_topo_error {} te_quant_topo_error {} (vs {})".format(
            lowest_te_quant_error_combo,
            np.round(
                sel_hp_som.quantization_error(tr_data)
                + sel_hp_som.topographic_error(tr_data),
                3,
            ),
            np.round(
                sel_hp_som.quantization_error(te_data)
                + sel_hp_som.topographic_error(te_data),
                3,
            ),
            np.round(lowest_te_quant_error, 3),
        )
    )

    # Visualizing after training
    sel_hp_som_trained_w = sel_hp_som.get_weights()
    sel_hp_nodes_labels, _, _ = classify(
        sel_hp_som, te_data, tr_data, tr_target_num, "majority_voting"
    )

    # Computing accuracies on all train and test samples
    log.info(">>>>> Final accuracies")
    _, _, _, _, _, _ = get_som_stats(
        sel_hp_som, tr_data, tr_target_num, te_data, te_target_num
    )

    visualize_samples_per_neuron(
        som_labels_map=sel_hp_som.labels_map(tr_data, tr_target_num.astype(str)),
        label_names=label_names,
        gt_labels=sel_hp_nodes_labels,
        colors=colors,
        file_name="som_multiclass_labeling_class_!_sel_hp_visualize_samples_per_neuron_tr.png",
    )
    visualize_samples_per_neuron(
        som_labels_map=sel_hp_som.labels_map(te_data, te_target_num.astype(str)),
        label_names=label_names,
        gt_labels=sel_hp_nodes_labels,
        colors=colors,
        file_name="som_multiclass_labeling_class_!_sel_hp_visualize_samples_per_neuron_te.png",
    )

    sel_hp_som_x_r = (
        np.min(sel_hp_som.get_euclidean_coordinates()[0]),
        np.max(sel_hp_som.get_euclidean_coordinates()[0]),
    )
    sel_hp_som_y_r = (
        np.min(sel_hp_som.get_euclidean_coordinates()[1]),
        np.max(sel_hp_som.get_euclidean_coordinates()[1]),
    )
    if top_c == "rectangular":
        visualize_rect_u_matrix(
            sel_hp_som.distance_map(),
            "som_multiclass_labeling_class_!_sel_hp_visualize_rect_u_matrix.png",
            sel_hp_som_x_r,
            sel_hp_som_y_r,
        )
    elif top_c == "hexagonal":
        visualize_hexag_u_matrix(
            sel_hp_som.get_euclidean_coordinates(),
            sel_hp_som.distance_map(),
            sel_hp_som_trained_w,
            "som_multiclass_labeling_class_!_sel_hp_visualize_hexag_u_matrix.png",
            sel_hp_som_x_r,
            sel_hp_som_y_r,
        )
    else:
        raise NotImplementedError

    if num_sliding_w == 1:
        visualize_feature_influence(
            sel_hp_som_trained_w,
            swarm_behaviour_feats_combo.names,
            file_name="som_multiclass_labeling_class_!_sel_hp_visualize_feat_influence.png",
        )
    else:
        visualize_time_series_feature_influence(
            sel_hp_som_trained_w,
            swarm_behaviour_feats_combo.names,
            file_name="som_multiclass_labeling_class_!_sel_hp_visualize_time_series_feat_influence.png",
            x_range=sel_hp_som_x_r,
            y_range=sel_hp_som_y_r,
        )

    visualize_prediction_trajectories(
        som_wn_fn=sel_hp_som.winner,
        som_n_coords=sel_hp_som.get_euclidean_coordinates(),
        te_data=te_data,
        te_target=te_target,
        num_te_samples_same_run=int(
            te_data.shape[0] // (num_behaviours * te_f_rand_idxs_shape0)
        ),
        file_name="som_multiclass_labeling_class_!_sel_hp_visualize_prediction_trajectories_run_{}.png",
    )

    # Export weights to .hdf5
    try:
        out_hdf5_file_name = "sel_hp_som_trained_weights.h5"
        with h5py.File(out_hdf5_file_name, "w") as hdf5_f:
            hdf5_f.create_dataset("my_data", data=sel_hp_som_trained_w)

        log.info(f"Saved HDF5 file '{out_hdf5_file_name}' w/ trained SOM weights.")
    except Exception as exception:
        log.info(f"{exception}")


if __name__ == "__main__":
    main()
