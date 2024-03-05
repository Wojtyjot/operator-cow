import sys
from typing import *

import dgl
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import wandb
from log_plots import plot_predictions
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
    vel = vel_bc[0].repeat(200, 1)
    print(f"vel = {vel}")
    print(f"vel_bc = {vel_bc}")
    x = torch.linspace(0, L, 200).to(device)
    t = torch.zeros(200, 1).to(device)
    x = x.reshape(-1, 1)
    t = t.reshape(-1, 1)
    out = torch.cat((x, t, vel), dim=-1)
    return out


def get_u0_v2(vel_bc: torch.Tensor, L: float, device: str) -> torch.Tensor:
    """
    Function creates initial condition from boundary conditions
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
    normalizer_up,
    normalizer_y,
    normalizer_x,
    metric,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Optimize input to the model inlet bc to match mesurements
    """
    # Set model to eval mode
    model_surrogate.eval()
    AE_model.eval()
    gs = g
    g = g.to(device)
    lambda_reg = lambda_reg
    metric = metric
    L_temp = normalizer_x.transform(g.ndata["x"], inverse=True)
    print(f"L_temp shape = {L_temp.shape}")
    L = L_temp[-1, 0]  #### NEEED to renormalize !!!!!
    print(f"L = {L}")

    theta = theta.to(device)

    theta = normalizer_up.transform(theta, inverse=True).squeeze()
    # setup optimizable parameters:
    SV_rec = np.random.uniform(100, 140)
    SV_rec = torch.Tensor([SV_rec]).to(device).requires_grad_()

    # initialize random bc
    u_bc = torch.rand(100, 1).to(device).requires_grad_()

    # initialize bc same as mesurement need to renormalize
    u_temp = normalizer_y.transform(g.ndata["y"], inverse=True)
    # u_bc = u_temp[100 * 100 : 100 * 101, 2].squeeze().to(device).requires_grad_()
    # print(u_bc.is_leaf)
    # print(SV_rec.is_leaf)
    # print(f"u_bc = {u_bc.shape}")

    u0_rec = u_bc[0].detach().repeat(200, 1).requires_grad_()

    # A0_rec

    u0_pred = get_u0_v2(u0_rec, L, device)

    print(f"u0_rec shape = {u0_rec.shape}")
    # u0_rec = u0_rec.requires_grad_()
    print(u0_rec.is_leaf)
    print(f"u0_pred shape = {u0_pred.shape}")

    # optimizer list of torch tensors to optimizer
    # in paper LBFGS

    optimizer = torch.optim.LBFGS(
        [u_bc, SV_rec, u0_rec], lr=1, line_search_fn="strong_wolfe"
    )

    loss_fn = nn.MSELoss()

    # max_steps = 1000
    # print(theta)
    # print(type(theta))
    # print(theta[:-1])
    SV_true = theta[-1]
    theta_const = theta[:-1].to(device)

    # Need to add eps as convergaence threshold
    for i in range(max_steps):
        print(f"step {i}")

        def closure():
            optimizer.zero_grad()
            gs = g

            # prepare data for forward pass

            theta_pred = torch.cat((theta_const, SV_rec), dim=0)
            # print(f"theta_pred befor norm {theta_pred.shape}")
            theta_pred = normalizer_up.transform(theta_pred, inverse=False).squeeze()
            # print(f"theta_pred after norm {theta_pred.shape}")

            bc = get_BC(u_bc, device)

            true_in_BC = inputs_f[0]
            p0 = inputs_f[1]  # holding constant
            u0 = inputs_f[2]
            a0 = inputs_f[3]  # holding constant

            u0_pred = get_u0_v2(u0_rec, L, device).unsqueeze(0)

            bc = bc.unsqueeze(0)
            p0 = p0.unsqueeze(0)
            # u0_pred = u0_pred.unsqueeze(0)
            a0 = a0.unsqueeze(0)
            inputs_f_pred = MultipleTensors([i for i in (bc, p0, u0_pred, a0)])

            # forward pass

            ### test dimensions
            # print(true_in_BC.shape == bc.shape)
            # print(u0.shape == u0_pred.shape)
            # print(theta.shape == theta_pred.shape)
            gs, theta_pred, inputs_f_pred = (
                gs.to(device),
                theta_pred.to(device),
                inputs_f_pred.to(device),
            )

            out = model_surrogate(gs, theta_pred.unsqueeze(0), inputs_f_pred)
            reg = AE_model(u_bc.T)
            # out [[p(x0, t0), a(x0, t0), u(x0, t0)]
            #      [p(x0, t1), a(x0, t1), u(x0, t1)]
            # ]
            # let mesurement point be at x100
            # compute loss
            mesurement = gs.ndata["y"][100 * 100 : 100 * 101, 2].squeeze()
            A0_mes = gs.ndata["y"][0::100, 1]

            # print(mesurement)
            # print(f"a0 out shape = {out[0::100, 1].shape}")
            # print(f"a0 init shape = {a0.shape}")
            # print(f"a0 init a0.squeeze()[:,2] = {a0.squeeze()[:,2]}")

            a0_loss = loss_fn(out[0::100, 1], A0_mes)  # ZNORMALIZOWAĆ !!!!
            mes_loss = loss_fn(out[100 * 100 : 100 * 101, 2].squeeze(), mesurement)
            reg_loss = loss_fn(reg, u_bc.T)

            loss = mes_loss + (lambda_reg * reg_loss) + a0_loss
            loss.backward()

            # backward pass
            # plot_predictions(out.detach().cpu().numpy(), g.ndata["y"].squeeze().detach().cpu().numpy(), "2137")

            # l2_domain, _, _ = metric(g, out, g.ndata["y"].squeeze())

            return loss

        optimizer.step(closure)
        theta_pred = torch.cat((theta_const, SV_rec), dim=0)
        # print(f"theta_pred befor norm {theta_pred.shape}")
        theta_pred = normalizer_up.transform(theta_pred, inverse=False).squeeze()
        # print(f"theta_pred after norm {theta_pred.shape}")

        bc = get_BC(u_bc, device)

        true_in_BC = inputs_f[0]
        p0 = inputs_f[1]  # holding constant
        u0 = inputs_f[2]
        a0 = inputs_f[3]  # holding constant

        u0_pred = get_u0_v2(u0_rec, L, device).unsqueeze(0)

        bc = bc.unsqueeze(0)
        p0 = p0.unsqueeze(0)
        # u0_pred = u0_pred.unsqueeze(0)
        a0 = a0.unsqueeze(0)
        inputs_f_pred = MultipleTensors([i for i in (bc, p0, u0_pred, a0)])

        # forward pass

        ### test dimensions
        # print(true_in_BC.shape == bc.shape)
        # print(u0.shape == u0_pred.shape)
        # print(theta.shape == theta_pred.shape)
        g, theta_pred, inputs_f_pred = (
            g.to(device),
            theta_pred.to(device),
            inputs_f_pred.to(device),
        )

        out = model_surrogate(g, theta_pred.unsqueeze(0), inputs_f_pred)
        reg = AE_model(u_bc.T)
        # out [[p(x0, t0), a(x0, t0), u(x0, t0)]
        #      [p(x0, t1), a(x0, t1), u(x0, t1)]
        # ]
        # let mesurement point be at x100
        # compute loss
        mesurement = g.ndata["y"][100 * 100 : 100 * 101, 2].squeeze()
        A0_mes = g.ndata["y"][0::100, 1]

        # print(mesurement)
        # print(f"a0 out shape = {out[0::100, 1].shape}")
        # print(f"a0 init shape = {a0.shape}")
        # print(f"a0 init a0.squeeze()[:,2] = {a0.squeeze()[:,2]}")

        a0_loss = loss_fn(out[0::100, 1], A0_mes)  # ZNORMALIZOWAĆ !!!!
        mes_loss = loss_fn(out[100 * 100 : 100 * 101, 2].squeeze(), mesurement)
        reg_loss = loss_fn(reg, u_bc.T)

        loss = mes_loss + (lambda_reg * reg_loss) + a0_loss

        # backward pass
        plot_predictions(
            out.detach().cpu().numpy(),
            g.ndata["y"].squeeze().detach().cpu().numpy(),
            "2137",
        )

        l2_domain, _, _ = metric(g, out, g.ndata["y"].squeeze())

        if i % 10 == 0:
            print(f"Step {i}, Loss {loss.item()}")
        wandb.log(
            {
                "Loss": loss.item(),
                "Mesurement Loss": mes_loss.item(),
                "Reg Loss": reg_loss.item(),
                "SV_rec": SV_rec.item(),
                "L2_whole domain": l2_domain.item(),
                "A0_loss": a0_loss.item(),
            }
        )

    return (u_bc, SV_rec, out, true_in_BC, SV_true)


def optimize_input_test_VANO(
    model_surrogate: nn.Module,
    VANO_model: nn.Module,
    g: dgl.DGLGraph,
    inputs_f: List[torch.Tensor],
    theta: torch.Tensor,
    lambda_reg: float,
    max_steps: int,
    device: str,
    normalizer_up,
    normalizer_y,
    normalizer_x,
    metric,
):
    # Set model to eval mode
    model_surrogate.eval()
    VANO_model.eval()
    gs = g
    g = g.to(device)
    lambda_reg = lambda_reg
    metric = metric
    L_temp = normalizer_x.transform(g.ndata["x"], inverse=True)
    # print(f"L_temp shape = {L_temp.shape}")
    L = L_temp[-1, 0]
    # print(f"L = {L}")

    theta = theta.to(device)

    condition = theta[:11]

    theta = normalizer_up.transform(theta, inverse=True).squeeze()
    # setup optimizable parameters:
    SV_rec = np.random.uniform(100, 140)
    SV_rec = torch.Tensor([SV_rec]).to(device).requires_grad_()

    # initialize random bc
    u_bc_latent = torch.randn(16, 1).to(device).requires_grad_()

    # initialize bc same as mesurement need to renormalize
    u_temp = normalizer_y.transform(g.ndata["y"], inverse=True)
    # u_bc = u_temp[100 * 100 : 100 * 101, 2].squeeze().to(device).requires_grad_()
    # print(u_bc.is_leaf)
    # print(SV_rec.is_leaf)
    # print(f"u_bc = {u_bc.shape}")

    # u0_rec = u_bc[0].detach().repeat(200,1).requires_grad_()

    # A0_rec

    # u0_pred = get_u0_v2(u0_rec, L, device)

    # print(f"u0_rec shape = {u0_rec.shape}")
    # u0_rec = u0_rec.requires_grad_()
    # print(u0_rec.is_leaf)
    # print(f"u0_pred shape = {u0_pred.shape}")

    # optimizer list of torch tensors to optimizer
    # in paper LBFGS

    optimizer = torch.optim.Adam([u_bc_latent, SV_rec], lr=1)

    loss_fn = nn.MSELoss()

    # max_steps = 1000
    # print(theta)
    # print(type(theta))
    # print(theta[:-1])
    SV_true = theta[-1]
    theta_const = theta[:-1].to(device)

    # Need to add eps as convergaence threshold
    for i in range(max_steps):
        print(f"step {i}")

        optimizer.zero_grad()
        gs = g

        # prepare data for forward pass

        theta_pred = torch.cat((theta_const, SV_rec), dim=0)
        # print(f"theta_pred befor norm {theta_pred.shape}")
        theta_pred = normalizer_up.transform(theta_pred, inverse=False).squeeze()
        # print(f"theta_pred after norm {theta_pred.shape}")

        u_bc_sampled = VANO_model.decode(
            u_bc_latent.reshape(1, 16), condition.reshape(1, 11)
        )

        bc = get_BC(u_bc_sampled, device)

        true_in_BC = inputs_f[0]
        p0 = inputs_f[1]  # holding constant
        u0 = inputs_f[2]
        a0 = inputs_f[3]  # holding constant

        # u0_pred = get_u0_v2(u0_rec, L, device).unsqueeze(0)

        bc = bc.unsqueeze(0)
        p0 = p0.unsqueeze(0)
        u0 = u0.unsqueeze(0)
        # u0_pred = u0_pred.unsqueeze(0)
        a0 = a0.unsqueeze(0)
        inputs_f_pred = MultipleTensors([i for i in (bc, p0, u0, a0)])

        # forward pass

        ### test dimensions
        # print(true_in_BC.shape == bc.shape)
        # print(u0.shape == u0_pred.shape)
        # print(theta.shape == theta_pred.shape)
        gs, theta_pred, inputs_f_pred = (
            gs.to(device),
            theta_pred.to(device),
            inputs_f_pred.to(device),
        )

        out = model_surrogate(gs, theta_pred.unsqueeze(0), inputs_f_pred)
        # reg = AE_model(u_bc.T)
        # out [[p(x0, t0), a(x0, t0), u(x0, t0)]
        #      [p(x0, t1), a(x0, t1), u(x0, t1)]
        # ]
        # let mesurement point be at x100
        # compute loss
        mesurement = gs.ndata["y"][100 * 100 : 100 * 101, 2].squeeze()
        A0_mes = gs.ndata["y"][0::100, 1]

        # print(mesurement)
        # print(f"a0 out shape = {out[0::100, 1].shape}")
        # print(f"a0 init shape = {a0.shape}")
        # print(f"a0 init a0.squeeze()[:,2] = {a0.squeeze()[:,2]}")

        a0_loss = loss_fn(out[0::100, 1], A0_mes)  # ZNORMALIZOWAĆ !!!!
        mes_loss = loss_fn(out[100 * 100 : 100 * 101, 2].squeeze(), mesurement)
        # reg_loss = loss_fn(reg, u_bc.T)

        loss = mes_loss + a0_loss
        loss.backward()

        # backward pass
        # plot_predictions(out.detach().cpu().numpy(), g.ndata["y"].squeeze().detach().cpu().numpy(), "2137")

        # l2_domain, _, _ = metric(g, out, g.ndata["y"].squeeze())

        optimizer.step()
        theta_pred = torch.cat((theta_const, SV_rec), dim=0)
        # print(f"theta_pred befor norm {theta_pred.shape}")
        theta_pred = normalizer_up.transform(theta_pred, inverse=False).squeeze()
        # print(f"theta_pred after norm {theta_pred.shape}")

        # backward pass
        plot_predictions(
            out.detach().cpu().numpy(),
            g.ndata["y"].squeeze().detach().cpu().numpy(),
            "2137",
        )

        l2_domain, _, _ = metric(g, out, g.ndata["y"].squeeze())

        if i % 10 == 0:
            print(f"Step {i}, Loss {loss.item()}")
        wandb.log(
            {
                "Loss": loss.item(),
                "Mesurement Loss": mes_loss.item(),
                # "Reg Loss": reg_loss.item(),
                "SV_rec": SV_rec.item(),
                "L2_whole domain": l2_domain.item(),
                "A0_loss": a0_loss.item(),
            }
        )

    return (u_bc_latent, SV_rec, out, true_in_BC, SV_true)
