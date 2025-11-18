#!/bin/bash

# RUN all experiments
HYDRA_FULL_ERROR=1 python3 -m swarm_hydra.my_experi.multiclass_class_experi_preprocessing features.selected_combo="Hauert2022" experiment_name="Hauert2022_sim"

HYDRA_FULL_ERROR=1 python3 -m swarm_hydra.my_experi.multiclass_class_experi_preprocessing features.selected_combo="LocalBoidsFeats" experiment_name="LocalBoidsFeats_sim"

HYDRA_FULL_ERROR=1 python3 -m swarm_hydra.my_experi.multiclass_class_experi_preprocessing features.selected_combo="Yang2023" experiment_name="Yang2023_sim"

HYDRA_FULL_ERROR=1 python3 -m swarm_hydra.my_experi.multiclass_class_experi_preprocessing features.selected_combo="Kuckling2023" experiment_name="Kuckling2023_sim"

