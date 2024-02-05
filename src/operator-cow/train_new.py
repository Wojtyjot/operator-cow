import os
import pickle
import sys

import numpy as np
import torch
import torch.nn as nn
from data_utils import COWDataset, MIODataLoader
from models.optimizer import Adam
from torch.optim.lr_scheduler import OneCycleLR
from utils.utils import get_num_params


def train(
    model: nn.Module,
    loss_func: nn.Module,
    metric_func: nn.Module,
    train_loader: MIODataLoader,
    val_loader: MIODataLoader,
    optimizer: nn.optim.Optimizer,
    lr_scheduler: nn.optim.lr_scheduler,
    epochs: int,
    device: torch.device,
    grad_clip: float,
    start_epoch: int = 0,
    print_freq: int = 20,
    model_save_path: str = "./data/checkpoints/",
):
    pass
