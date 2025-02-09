

# Reconstruction of Cerebral Hemodynamics from Sparse Data Using Neural Operator Transformers 
## Overview


This is code repository for paper titled "Reconstruction of Cerebral Hemodynamics from Sparse Data Using Neural Operator Transformers"
It features scripts for reproduing results described in paper. 







## Getting Started

### Code and development environment

Dependencies are provieded in file requirements.txt

  We ran our experiments on Nvidia RTX A6000 Ada GPU
  To run them, you should have at least 48 GB of VRAM avaliable on GPU
  To run scripts on smaller GPU's loaders and batching may need to be changed.

### Data

Data for reproducing validation results will be meade avaliable on mendeley upon publication of manuscript.
Training and testing datasets will be made avaliable upon resonable request as they are too large to upload directly to mendely

### Logging and tracking experiments

We use [Weights & Biases](https://wandb.ai/site) to log and track our experiments.
If you're logged in, your default entity will be used (a fixed entity is not set in the config),
and you can set another entity with the `WANDB_ENTITY` environment variable.
Otherwise, the runs will be anonymous (you don't need to be logged in).

## Reproduction and Experimentation

### Reproducing our results
To reproduce our results use model checkpoints provided in `data/checkpoints/` direcotry.
Edit data paths in scripts stored in the `configs/` directory.
To train your own model update config files and run
```python
python run_experiment.py
```
To train generative model run
```python
python VANO_experiment.py
```
To run inverse experiment run:
```python
python Multi_inverse.py
```
To run invese and save estimated Windkessel parameters run

```python
python Multi_Windkessel.py
```
### Experiment with different configurations

The default configuration for each script is stored in the `configs/` directory.
They are managed by [Hydra](https://hydra.cc/docs/intro/).
You can experiment with different configurations by passing the relevant arguments.

## Repository structure

Below, we give a description of the main files and directories in this repository.

```
 └─── data/                           # Data directory
    ├── checkpoints/                 # Model checkpoints
    ├── README.md                    # Data documentation
    └── joints.csv                   # File encoding CoW topography
└─── src/                           # Source code
    └── operatorcow/                # Main package
        ├── configs/                 # Configuration files
        │   ├── override/           # Override configurations
        │   ├── GANO.yaml           # GANO model config
        │   ├── inverse_full_new_hyper.yaml  # Inverse problem config
        │   ├── Plots.yaml          # Plotting configuration
        │   ├── VANO_exp.yaml       # VANO experiment config
        │   ├── setup.yaml          # Setup configuration
        │   ├── sweep_config.yaml   # Parameter sweep config
        │   └── test_exp.yaml       # Test experiment config
        ├── inverse/                # Inverse problem implementations
        │   ├── COW.py             # Circle of Willis implementation
        │   └── Find_RCR.py        # Boundary condition reconstruction
        ├── models/                 # Model implementations
        │   ├── VANO.py            # VANO model
        │   ├── __init__.py        # Package initialization
        │   ├── ae.py              # Autoencoder implementation
        │   ├── cgpt.py            # Model implementation
        │   ├── mgpt.py            # Model implementation
        │   ├── mlp.py             # Multi-layer perceptron
        │   ├── mmgpt.py           # GNOT implementation
        │   └── optimizer.py       # Optimization utilities
        └── utils/                  # Utility functions
            ├── Create_GNOT_plots.py  # Plotting utilities for GNOT
            ├── Inverse_full.py     # Inverse problem utilities
            ├── Inverse_sweep.py    # Parameter sweep utilities
            ├── Multi_Windkessel.py # Inverse reconstruction with Windkessel estimation
            ├── Multi_inverse.py    # Inverse reconstruction
            ├── VANO_experiment.py  # VANO experiment utilities
            ├── __init__.py        # Utils initialization
            ├── args.py            # Argument parsing
            ├── data_utils.py      # Data processing utilities
            ├── init.py            # Initialization utilities
            ├── log_plots.py       # Logging and plotting
            ├── run_experiment.py  # GNOT training experiment
            ├── train.py           # Training utilities
            └── train_new.py       # Updated training utilities
```


