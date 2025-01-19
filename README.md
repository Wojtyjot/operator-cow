

# Reconstruction of Cerebral Hemodynamics from Sparse Data Using Neural Operator Transformers 
## Overview


This is code repository for paper titled "Reconstruction of Cerebral Hemodynamics from Sparse Data Using Neural Operator Transformers"
It features scripts for reproduing results described in paper. 







## Getting Started

### Code and development environment

Dependencies are provieded in file requirements.txt

  We ran our experiments on Nvidia RTX A6000 Ada GPU
  To run them, you should have at least 48 GB of VRAM avaliable on GPU
  To run scripts on smaller GPU's loaders may need to be changed.

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


We provide scripts to reproduce our work in the `reproducibility-scripts/` directory.
It has a README at its root describing which scripts reproduce which experiments.

### Experiment with different configurations

The default configuration for each script is stored in the `configs/` directory.
They are managed by [Hydra](https://hydra.cc/docs/intro/).
You can experiment with different configurations by passing the relevant arguments.
You can get examples of how to do so in the `reproducibility-scripts/` directory.

### Using trained models and experimenting with results

Moreover, we make our trained models available.
You can follow the instructions in `outputs/README.md` to download and use them.

## Repository structure

Below, we give a description of the main files and directories in this repository.

```
 └─── src/                              # Source code.
    └── operator-cow           # Our package.
        ├── configs/                    # Hydra configuration files.
        └── template_experiment.py      # A template experiment.
```


