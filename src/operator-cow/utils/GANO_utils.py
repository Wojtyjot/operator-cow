from typing import Tuple

import numpy as np
import torch
import torch.nn as nn


def compute_MFV(
    velocity: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute mean flow velocity of generated samples

    MFV = min(flow, dim=1) + (max(flow, dim=1) - min(flow, dim=1)) / 3

    Returns:
    mean: torch.Tensor
        Mean of MFV in generated samples

    var: torch.Tensor
        Variance of MFV in generated samples

    MFV: torch.Tensor
        Mean flow velocity of generated samples
    """
    # flow size = (batch_szie/num_generated samples for testing, 100, 1)

    min = torch.min(velocity, dim=1)[0]
    max = torch.max(velocity, dim=1)[0]
    MFV = min + (max - min) / 3
    mean = torch.mean(MFV, dim=0)
    var = torch.var(MFV, dim=0)
    return mean, var, MFV


def compute_PI(velocity: torch.Tensor, MFV: torch.Tensor) -> torch.Tensor:
    """
    Computes Pulsaruity Index (PI) of generated samples

    PI = (max(flow, dim=1) - min(flow, dim=1)) / MFV

    """
    # flow size = (batch_szie/num_generated samples for testing, 100, 1)
    min = torch.min(velocity, dim=1)[0]
    max = torch.max(velocity, dim=1)[0]
    PI = (max - min) / MFV
    return PI


def compute_statistics(velocity: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute MFV and PI of generated samples

    velocity size = (8,100, 1)
    """
    MFV = (
        torch.min(velocity, dim=1)[0]
        + (torch.max(velocity, dim=1)[0] - torch.min(velocity, dim=1)[0]) / 3
    )
    PI = (torch.max(velocity, dim=1)[0] - torch.min(velocity, dim=1)[0]) / MFV

    return MFV, PI
