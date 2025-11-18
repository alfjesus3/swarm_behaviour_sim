#!/bin/bash

# RUN all experiments
uv run tests/test_flocking.py \
  'seeds_per_exp_rep=[1]' \
  behaviours.flocking.steps=1000 \
  behaviours.flocking.boid_count=125 \
  behaviours.flocking.storing_data=False \
  behaviours.flocking.walls_b=False \
  'behaviours.flocking.seed=${seeds_per_exp_rep[0]}' \
  'behaviours.class_m=${behaviours.flocking}'

uv run tests/test_aggregation.py \
  'seeds_per_exp_rep=[1]' \
  behaviours.aggregation.steps=2000 \
  behaviours.aggregation.boid_count=125 \
  behaviours.aggregation.storing_data=False \
  behaviours.aggregation.walls_b=True \
  'behaviours.aggregation.seed=${seeds_per_exp_rep[0]}' \
  'behaviours.class_m=${behaviours.aggregation}'

uv run tests/test_dispersion.py \
  'seeds_per_exp_rep=[1]' \
  behaviours.dispersion.steps=2000 \
  behaviours.dispersion.boid_count=125 \
  behaviours.dispersion.storing_data=False \
  behaviours.dispersion.walls_b=True \
  'behaviours.dispersion.seed=${seeds_per_exp_rep[0]}' \
  'behaviours.class_m=${behaviours.dispersion}'

uv run tests/test_random_walk.py \
  'seeds_per_exp_rep=[1]' \
  behaviours.random_walk.steps=2000 \
  behaviours.random_walk.boid_count=125 \
  behaviours.random_walk.storing_data=False \
  behaviours.random_walk.walls_b=True \
  'behaviours.random_walk.seed=${seeds_per_exp_rep[0]}' \
  'behaviours.class_m=${behaviours.random_walk}'

uv run tests/test_metrics.py
