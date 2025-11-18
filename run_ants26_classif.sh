#!/bin/bash

# RUN all experiments
HYDRA_FULL_ERROR=1 python3 -m swarm_hydra.my_experi.som_multiclass_labeling_class \
  features.selected_combo="Yang2023" \
  'seeds_per_exp_rep=[1]' \
  experiment_name="Yang2023_v1"

HYDRA_FULL_ERROR=1 python3 -m swarm_hydra.my_experi.som_multiclass_labeling_class \
  features.selected_combo="Yang2023" \
  'seeds_per_exp_rep=[2]' \
  experiment_name="Yang2023_v2"

HYDRA_FULL_ERROR=1 python3 -m swarm_hydra.my_experi.som_multiclass_labeling_class \
  features.selected_combo="Hauert2022" \
  'seeds_per_exp_rep=[1]' \
  experiment_name="ants26_Hauert2022_v1"

HYDRA_FULL_ERROR=1 python3 -m swarm_hydra.my_experi.som_multiclass_labeling_class \
  features.selected_combo="Hauert2022" \
  'seeds_per_exp_rep=[2]' \
  experiment_name="ants26_Hauert2022_v2"

HYDRA_FULL_ERROR=1 python3 -m swarm_hydra.my_experi.som_multiclass_labeling_class \
  features.selected_combo="Kuckling2023" \
  'seeds_per_exp_rep=[1]' \
  experiment_name="Kuckling2023_v1"

HYDRA_FULL_ERROR=1 python3 -m swarm_hydra.my_experi.som_multiclass_labeling_class \
  features.selected_combo="Kuckling2023" \
  'seeds_per_exp_rep=[2]' \
  experiment_name="Kuckling2023_v2"

HYDRA_FULL_ERROR=1 python3 -m swarm_hydra.my_experi.som_multiclass_labeling_class \
  features.selected_combo="LocalBoidsFeats" \
  'seeds_per_exp_rep=[1]' \
  experiment_name="ants26_LocalBoidsFeats_v1"

HYDRA_FULL_ERROR=1 python3 -m swarm_hydra.my_experi.som_multiclass_labeling_class \
  features.selected_combo="LocalBoidsFeats" \
  'seeds_per_exp_rep=[2]' \
  experiment_name="ants26_LocalBoidsFeats_v2"

HYDRA_FULL_ERROR=1 python3 -m swarm_hydra.my_experi.rand_forest_multiclass_class \
  features.selected_combo="Yang2023" \
  'seeds_per_exp_rep=[1]' \
  experiment_name="Yang2023_rf_v1"

HYDRA_FULL_ERROR=1 python3 -m swarm_hydra.my_experi.rand_forest_multiclass_class \
  features.selected_combo="Yang2023" \
  'seeds_per_exp_rep=[2]' \
  experiment_name="Yang2023_rf_v2"

HYDRA_FULL_ERROR=1 python3 -m swarm_hydra.my_experi.rand_forest_multiclass_class \
  features.selected_combo="Hauert2022" \
  'seeds_per_exp_rep=[1]' \
  experiment_name="ants26_Hauert2022_rf_v1"

HYDRA_FULL_ERROR=1 python3 -m swarm_hydra.my_experi.rand_forest_multiclass_class \
  features.selected_combo="Hauert2022" \
  'seeds_per_exp_rep=[2]' \
  experiment_name="ants26_Hauert2022_rf_v2"

HYDRA_FULL_ERROR=1 python3 -m swarm_hydra.my_experi.rand_forest_multiclass_class \
  features.selected_combo="Kuckling2023" \
  'seeds_per_exp_rep=[1]' \
  experiment_name="Kuckling2023_rf_v1"

HYDRA_FULL_ERROR=1 python3 -m swarm_hydra.my_experi.rand_forest_multiclass_class \
  features.selected_combo="Kuckling2023" \
  'seeds_per_exp_rep=[2]' \
  experiment_name="Kuckling2023_rf_v2"

HYDRA_FULL_ERROR=1 python3 -m swarm_hydra.my_experi.rand_forest_multiclass_class \
  features.selected_combo="LocalBoidsFeats" \
  'seeds_per_exp_rep=[1]' \
  experiment_name="ants26_LocalBoidsFeats_rf_v1"

HYDRA_FULL_ERROR=1 python3 -m swarm_hydra.my_experi.rand_forest_multiclass_class \
  features.selected_combo="LocalBoidsFeats" \
  'seeds_per_exp_rep=[2]' \
  experiment_name="ants26_LocalBoidsFeats_rf_v2"
