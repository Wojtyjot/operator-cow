from typing import *

import dgl
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import wandb
from utils.utils import MultipleTensors


def get_BC(velocity: torch.Tensor, device: str) -> torch.Tensor:
    """
    Function computes inflow boundary conditon from velocity array

    BC in form [
        [x0, t0, u(x0, t0)],
        [x0, t1, u(x0, t1)],
        ...
    ]
    dim = nt X 3

    Parameters
    ----------
    velocity : torch.Tensor
        Velocity array (nx, nt)
    """
    x = torch.zeros((100, 1)).to(device)
    t = torch.linspace(0, 1, 100).to(device)
    x = x.reshape(-1, 1)
    t = t.reshape(-1, 1)
    out = torch.cat((x, t, velocity.reshape(-1, 1)), dim=-1)
    return out


def get_u0(vel_bc: torch.Tensor, L: float, device: str) -> torch.Tensor:
    """
    Function creates initial condition for velocity
    """
    x = torch.linspace(0, L, 200).to(device)
    t = torch.zeros(200, 1).to(device)
    x = x.reshape(-1, 1)
    t = t.reshape(-1, 1)
    out = torch.cat((x, t, vel_bc), dim=-1)
    return out


def optimize_input_test(
    model_surrogate: nn.Module,
    AE_model: nn.Module,
    g: dgl.DGLGraph,
    inputs_f: List[torch.Tensor],
    theta: torch.Tensor,
    lambda_reg: float,
    max_steps: int,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Optimize input to the model inlet bc to match mesurements
    """
    # Set model to eval mode
    model_surrogate.eval()
    AE_model.eval()
    L = g.ndata["x"][200, 0]
    # setup optimizable parameters:
    SV_rec = np.random.uniform(70, 140)
    SV_rec = torch.Tensor(SV_rec, requires_grad=True).to(device)

    # initialize random bc
    u_bc = torch.rand(100, 1, requires_grad=True).to(device)

    # optimizer list of torch tensors to optimizer

    optimizer = torch.optim.Adam([u_bc, SV_rec], lr=0.1)

    loss_fn = nn.MSELoss()

    # max_steps = 1000
    SV_true = theta[-1]
    theta_const = theta[:-1].to(device)

    # Need to add eps as convergaence threshold
    for i in range(max_steps):
        optimizer.zero_grad()

        # prepare data for forward pass

        theta_pred = torch.cat((theta_const, SV_rec), dim=0)

        bc = get_BC(u_bc, device)

        true_in_BC = inputs_f[0]
        p0 = inputs_f[1]  # holding constant
        u0 = inputs_f[2]
        a0 = inputs_f[3]  # holding constant

        u0_pred = get_u0(u_bc, L, device)

        inputs_f_pred = MultipleTensors([i for i in (bc, p0, u0_pred, a0)])

        # forward pass
        out = model_surrogate(g, inputs_f_pred, theta_pred)
        reg = AE_model(u_bc)
        # out [[p(x0, t0), a(x0, t0), u(x0, t0)]
        #      [p(x0, t1), a(x0, t1), u(x0, t1)]
        # ]
        # let mesurement point be at x100
        # compute loss
        mesurement = g.ndata["y"][100 * 100 : 100 * 101, 2].squeeze()

        mes_loss = loss_fn(out[100 * 100 : 100 * 101, 2].squeeze(), mesurement)
        reg_loss = loss_fn(reg, u_bc)

        loss = mes_loss + (lambda_reg * reg_loss)

        # backward pass
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            print(f"Step {i}, Loss {loss.item()}")
        wandb.log(
            {
                "Loss": loss.item(),
                "Mesurement Loss": mes_loss.item(),
                "Reg Loss": reg_loss.item(),
                "SV_rec": SV_rec.item(),
            }
        )

    return (u_bc, SV_rec, out, true_in_BC, SV_true)
