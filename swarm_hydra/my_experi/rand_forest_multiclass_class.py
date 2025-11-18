import hydra
from omegaconf import DictConfig, OmegaConf
import logging

import numpy as np
import sys
import os
import subprocess

import pickle

from minisom import MiniSom
import minisom
import matplotlib.pyplot as plt
import matplotlib
import itertools
import sklearn.ensemble
from sklearn.tree import export_graphviz
from tqdm import tqdm
import h5py
import gc
from typing import Callable
from collections import defaultdict, Counter
from copy import deepcopy

from swarm_hydra.entry_point import *
from swarm_hydra.my_experi.multiclass_class_experi import *

import sklearn

# A logger for this file
log = logging.getLogger(__name__)


def export_rf(rf: sklearn.ensemble.RandomForestClassifier, rf_path: str) -> None:
    """"""
    with open(rf_path, "wb") as f_rf:
        pickle.dump(rf, f_rf)


def load_rf(rf_path: str) -> sklearn.ensemble.RandomForestClassifier:
    """"""
    rf = None
    with open(rf_path, "rb") as f_rf:
        rf = pickle.load(f_rf)
    return rf


def export_rf_tree_to_png(
    model, tree_index, feature_names, class_names, output_filename="tree.png", dpi=600
):
    """
    Export a specific tree from a RandomForest model to a PNG image using Graphviz.

    Parameters:
        model: Trained RandomForestClassifier or RandomForestRegressor
        tree_index (int): Index of the tree within the forest (e.g., 0 to n_estimators-1)
        feature_names (list): List of feature names
        class_names (list): List of class names (for classifiers)
        output_filename (str): Path to output .png file
        dpi (int): DPI for the output image
    """

    # Extract the estimator
    estimator = model.estimators_[tree_index]

    # Temporary DOT filename
    dot_filename = output_filename.replace(".png", ".dot")

    # Export to .dot
    export_graphviz(
        estimator,
        out_file=dot_filename,
        feature_names=feature_names,
        class_names=class_names,
        rounded=True,
        proportion=False,
        precision=2,
        filled=True,
    )

    # Convert to PNG using Graphviz
    try:
        subprocess.run(
            ["dot", "-Tpng", dot_filename, "-o", output_filename, f"-Gdpi={dpi}"],
            check=True,
        )
        print(f"Tree exported to {output_filename}")
    except subprocess.CalledProcessError as e:
        print(f"Error converting DOT to PNG: {e}")

    # Optional: clean up .dot file
    os.remove(dot_filename)


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
        _,
        _,
        _,
        _,
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
        "tr_data.shape {} tr_target.shape {} te_data.shape {} te_target.shape {}".format(
            tr_data.shape,
            tr_target.shape,
            te_data.shape,
            te_target.shape,
        )
    )

    ## Hyperparameter optimization
    param_grid = {
        "n_estimators": [
            int(x) for x in np.linspace(100, 500, num=3)
        ],  # number of trees in the random forest
        # criterion: Literal['gini', 'entropy', 'log_loss'] = "gini",
        "max_depth": [
            int(x) for x in np.linspace(10, 50, num=3)
        ],  # maximum number of levels allowed in each decision tree
        "min_samples_split": [2, 6, 10],  # minimum sample number to split a node,
        "min_samples_leaf": [
            1,
            3,
            4,
        ],  # minimum sample number that can be stored in a leaf node,
        # min_weight_fraction_leaf: Float = 0,
        # max_leaf_nodes: Int | None = None,
        # min_impurity_decrease: Float = 0,
        "max_features": [
            "log2",
            "sqrt",
        ],  # number of features in consideration at every split
        "bootstrap": [True, False],  # method used to sample data points
        # oob_score: bool = False,
        # n_jobs: Int | None = None,
        "random_state": [cfg.behaviours.class_m.seed],
        # verbose: Int = 0,
        # warm_start: bool = False,
        # class_weight: Mapping | Sequence[Mapping] | Literal['balanced', 'balanced_subsample'] | None = None,
        # ccp_alpha: float = 0,
        # max_samples: float | None = None
    }
    rf = sklearn.ensemble.RandomForestClassifier()
    rf_fit = sklearn.model_selection.RandomizedSearchCV(
        rf,
        param_grid,
        n_iter=10,
        cv=3,
        verbose=2,
        random_state=cfg.behaviours.class_m.seed,
        n_jobs=3,  # -1,  #
    )
    rf_fit.fit(
        tr_data if use_all_data else tr_data[tr_idxs_sub],
        tr_target_num if use_all_data else tr_target_num[tr_idxs_sub],
    )
    sel_hp_rf_param = rf_fit.best_params_
    log.info("The selected hp combo is {}".format(sel_hp_rf_param))

    # sel_hp_rf_param = {
    #     "random_state": 0,
    #     "n_estimators": 100,
    #     "min_samples_split": 2,
    #     "min_samples_leaf": 1,
    #     "max_features": "sqrt",
    #     "max_depth": 120,
    #     "bootstrap": False,
    # }

    # Retraining and testing with the selected hp config
    sel_hp_rf = sklearn.ensemble.RandomForestClassifier(
        n_estimators=sel_hp_rf_param["n_estimators"],
        min_samples_split=sel_hp_rf_param["min_samples_split"],
        min_samples_leaf=sel_hp_rf_param["min_samples_leaf"],
        max_features=sel_hp_rf_param["max_features"],
        max_depth=sel_hp_rf_param["max_depth"],
        bootstrap=sel_hp_rf_param["bootstrap"],
        random_state=sel_hp_rf_param["random_state"],
    )
    sel_hp_rf.fit(
        tr_data if use_all_data else tr_data[tr_idxs_sub],
        tr_target_num if use_all_data else tr_target_num[tr_idxs_sub],
    )
    # Classify train and test data
    tr_preds = sel_hp_rf.predict(tr_data)
    accuracy = sklearn.metrics.accuracy_score(tr_target_num, tr_preds)
    jaccard_score = sklearn.metrics.jaccard_score(
        tr_target_num, tr_preds, average="macro"
    )
    precision_score = sklearn.metrics.precision_score(
        tr_target_num,
        tr_preds,
        labels=[i for i in range(tr_target_num.shape[0])],
        average="macro",
    )
    recall_score = sklearn.metrics.recall_score(
        tr_target_num,
        tr_preds,
        labels=[i for i in range(tr_target_num.shape[0])],
        average="macro",
    )
    log.info(
        "Train acc {} jac {} prec {} rec {}".format(
            np.round(accuracy, 3),
            np.round(jaccard_score, 3),
            np.round(precision_score, 3),
            np.round(recall_score, 3),
        )
    )
    te_preds = sel_hp_rf.predict(te_data)
    accuracy = sklearn.metrics.accuracy_score(te_target_num, te_preds)
    jaccard_score = sklearn.metrics.jaccard_score(
        te_target_num, te_preds, average="macro"
    )
    precision_score = sklearn.metrics.precision_score(
        te_target_num,
        te_preds,
        labels=[i for i in range(te_target_num.shape[0])],
        average="macro",
    )
    recall_score = sklearn.metrics.recall_score(
        te_target_num,
        te_preds,
        labels=[i for i in range(te_target_num.shape[0])],
        average="macro",
    )
    log.info(
        "Test acc {} jac {} prec {} rec {}".format(
            np.round(accuracy, 3),
            np.round(jaccard_score, 3),
            np.round(precision_score, 3),
            np.round(recall_score, 3),
        )
    )

    # Exporting the RF
    export_rf(sel_hp_rf, "random_forest_multiclass_class_!_rf.pkl")
    # And the final tree structure for inspection
    all_feature_names = []
    for name in swarm_behaviour_feats_combo.names:
        for i in range(num_sliding_w):
            all_feature_names.append(name + f"{i+1}")
    export_rf_tree_to_png(
        model=sel_hp_rf,
        tree_index=5,
        feature_names=all_feature_names,
        class_names=label_names,
        output_filename="random_forest_multiclass_class_!_sel_hp_rf_structure.png",
        dpi=600,
    )


if __name__ == "__main__":
    main()
