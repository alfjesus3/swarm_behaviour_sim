#!/bin/bash

# RUN all experiments
python3 -m swarm_behaviour_sim.behaviours.generate_data \
  'seeds_per_exp_rep=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49]' \
  'behaviours.class_m=${behaviours.flocking}' \
  behaviours.flocking.steps=1500 \
  behaviours.flocking.boid_count=40 \
  behaviours.flocking.storing_data=True \
  behaviours.flocking.walls_b=True \
  experiment_name="dt_gen"

python3 -m swarm_behaviour_sim.behaviours.generate_data \
  'seeds_per_exp_rep=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49]' \
  'behaviours.class_m=${behaviours.flocking}' \
  behaviours.flocking.steps=1500 \
  behaviours.flocking.boid_count=40 \
  behaviours.flocking.storing_data=True \
  behaviours.flocking.walls_b=False \
  experiment_name="dt_gen"

python3 -m swarm_behaviour_sim.behaviours.generate_data \
  'seeds_per_exp_rep=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49]' \
  'behaviours.class_m=${behaviours.flocking}' \
  behaviours.flocking.steps=1500 \
  behaviours.flocking.boid_count=30 \
  behaviours.flocking.storing_data=True \
  behaviours.flocking.walls_b=True \
  experiment_name="dt_gen"

python3 -m swarm_behaviour_sim.behaviours.generate_data \
  'seeds_per_exp_rep=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49]' \
  'behaviours.class_m=${behaviours.aggregation}' \
  behaviours.aggregation.steps=1500 \
  behaviours.aggregation.boid_count=40 \
  behaviours.aggregation.storing_data=True \
  behaviours.aggregation.walls_b=True \
  experiment_name="dt_gen"

python3 -m swarm_behaviour_sim.behaviours.generate_data \
  'seeds_per_exp_rep=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49]' \
  'behaviours.class_m=${behaviours.aggregation}' \
  behaviours.aggregation.steps=1500 \
  behaviours.aggregation.boid_count=40 \
  behaviours.aggregation.storing_data=True \
  behaviours.aggregation.walls_b=False \
  experiment_name="dt_gen"

python3 -m swarm_behaviour_sim.behaviours.generate_data \
  'seeds_per_exp_rep=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49]' \
  'behaviours.class_m=${behaviours.aggregation}' \
  behaviours.aggregation.steps=1500 \
  behaviours.aggregation.boid_count=30 \
  behaviours.aggregation.storing_data=True \
  behaviours.aggregation.walls_b=True \
  experiment_name="dt_gen"

python3 -m swarm_behaviour_sim.behaviours.generate_data \
  'seeds_per_exp_rep=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49]' \
  'behaviours.class_m=${behaviours.dispersion}' \
  behaviours.dispersion.steps=1500 \
  behaviours.dispersion.boid_count=40 \
  behaviours.dispersion.storing_data=True \
  behaviours.dispersion.walls_b=True \
  experiment_name="dt_gen"

python3 -m swarm_behaviour_sim.behaviours.generate_data \
  'seeds_per_exp_rep=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49]' \
  'behaviours.class_m=${behaviours.dispersion}' \
  behaviours.dispersion.steps=1500 \
  behaviours.dispersion.boid_count=40 \
  behaviours.dispersion.storing_data=True \
  behaviours.dispersion.walls_b=False \
  experiment_name="dt_gen"

python3 -m swarm_behaviour_sim.behaviours.generate_data \
  'seeds_per_exp_rep=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49]' \
  'behaviours.class_m=${behaviours.dispersion}' \
  behaviours.dispersion.steps=1500 \
  behaviours.dispersion.boid_count=30 \
  behaviours.dispersion.storing_data=True \
  behaviours.dispersion.walls_b=True \
  experiment_name="dt_gen"

python3 -m swarm_behaviour_sim.behaviours.generate_data \
  'seeds_per_exp_rep=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49]' \
  'behaviours.class_m=${behaviours.random_walk}' \
  behaviours.random_walk.steps=1500 \
  behaviours.random_walk.boid_count=40 \
  behaviours.random_walk.storing_data=True \
  behaviours.random_walk.walls_b=True \
  experiment_name="dt_gen"

python3 -m swarm_behaviour_sim.behaviours.generate_data \
  'seeds_per_exp_rep=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49]' \
  'behaviours.class_m=${behaviours.random_walk}' \
  behaviours.random_walk.steps=1500 \
  behaviours.random_walk.boid_count=40 \
  behaviours.random_walk.storing_data=True \
  behaviours.random_walk.walls_b=False \
  experiment_name="dt_gen"

python3 -m swarm_behaviour_sim.behaviours.generate_data \
  'seeds_per_exp_rep=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49]' \
  'behaviours.class_m=${behaviours.random_walk}' \
  behaviours.random_walk.steps=1500 \
  behaviours.random_walk.boid_count=30 \
  behaviours.random_walk.storing_data=True \
  behaviours.random_walk.walls_b=True \
  experiment_name="dt_gen"
