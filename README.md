# swarm_behaviour_sim
This package contains the code to generate the point mass simulation (under `swarm_hydra/behaviours/`) of Reynolds flocking, Vicsek flocking, aggregation, dispersion, ballistic random walk and Brownian random walk. It provides the logic to compare the similarity between behaviours and classify them in `swarm/hydra/my_experi`.


## Installation
### Install from Source (GitHub)

```bash
git clone https://github.com/alfjesus3/swarm_behaviour_sim
cd swarm_behaviour_sim
pip install -e .
```

## Structure
```
├── swarm_hydra
│   ├── entry_point.py
│   ├── __init__.py
│   ├── configs
│   │   ├── config.yaml
|   |   └── **/*.yaml
│   ├── behaviours
│   │   ├── __init__.py
│   │   ├── aggregation.py
│   │   ├── dispersion.py
│   │   ├── flocking.py
│   │   ├── generate_data.py
│   │   ├── potential_field.py
│   │   ├── random_walk.py
|   |   └── utils_behaviours.py
│   ├── metrics
│   │   ├── README.md
│   │   ├── __init__.py
│   │   ├── compute_metrics.py
│   │   ├── kinematic_tree.py
│   │   ├── proba_metrics.py
│   │   ├── spatial_metrics.py
│   │   ├── swarm_features_combos.py
│   │   ├── temporal_metrics.py
|   |   └── utils_metrics.py
│   ├── my_experi
│   │   ├── __init__.py
│   │   ├── multiclass_class_experi.py
│   │   ├── multiclass_class_experi_preprocessing.py
│   │   ├── rand_forest_multiclass_class.py
|   |   └──  som_multiclass_labeling_class.py
├── tests
│   ├── test_*.py
|   └── config.yaml
├── .gitignore
├── .python-version
├── LICENSE
├── README.md
├── generate_data.sh
├── pyproject.toml
├── run_ants26_classif.sh
├── run_ants26_sim.sh
├── run_tests.sh
└──uv.lock
```

## Scripts

- the `generate_data.sh` generates the collective behaviours dataset;
- the `run_ants26_sim.sh` generates the similarity assessment results from the ANTS26 manuscript;
- the `run_ants26_classif.sh` generates the classification results from the ANTS26 manuscript;


## Citing
When using `swarm_behaviour_sim` in an academic work please cite our publication using the following Bibtex citation:
```
```