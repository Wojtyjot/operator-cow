import os
import pickle
import sys

import numpy as np
import torch
import torch.nn as nn
import wandb
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
    device: str,
    grad_clip: float,
    start_epoch: int = 0,
    print_freq: int = 20,
    model_save_path: str = "../../data/checkpoints/",
    model_name: str = "model.pt",
    result_name: str = "results.pt",
):
    loss_train = []
    loss_val = []
    loss_epoch = []
    lr_history = []
    it = 0
    best_val_metric = np.inf

    for epoch in range(epochs):
        model.tarin()
        torch.cuda.empty_cache()

        for batch in train_loader:
            loss = train_batch(
                model=model,
                loss_func=loss_func,
                data=batch,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                device=device,
                grad_clip=grad_clip,
            )
            wandb.log(
                {
                    "L2loss": loss[0],
                    "lr": optimizer.param_groups[0]["lr"],
                }
            )
            loss = np.array(loss)
            loss_epoch.append(loss)
            lr_history.append(optimizer.param_groups[0]["lr"])
            it += 1
            log = f"epoch: [{epoch +1}/{epochs}]"
            log += f"loss: {loss[0]:.4f}"

            if it % print_freq == 0:
                print(log)

        val_result = validate_epoch(
            model=model,
            metric_func=metric_func,
            valid_loader=val_loader,
            device=device,
        )
        wandb.log({"val_L2_loss": val_result["metric"].mean()})

        val_metric = val_result["metric"].sum()
        loss_val.append(val_metric["metric"])

        if val_metric < best_val_metric:
            best_val_metric = val_metric
            best_val_epoch = epoch
            torch.save(
                model.state_dict(),
                os.path.join(model_save_path, model_name),
            )

        result = dict(
            best_val_epoch=best_val_epoch,
            best_val_metric=best_val_metric,
            loss_train=np.asarray(loss_train),
            loss_val=np.asarray(loss_val),
            optimizer_state=optimizer.state_dict(),
        )
        pickle.dump(result, open(os.path.join(model_save_path, result_name), "wb"))

    return result


def train_batch(
    model: nn.Module,
    loss_func: nn.Module,
    data: list,
    optimizer: nn.optim.Optimizer,
    lr_scheduler: nn.optim.lr_scheduler,
    device: str,
    grad_clip: float,
):
    optimizer.zero_grad()
    g, u_p, g_u = data

    g, u_p, g_u = g.to(device), u_p.to(device), g_u.to(device)

    out = model(g, u_p, g_u)

    y_pred, y = out.squeeze(), g.ndata["y"].squeeze()

    loss, reg, _ = loss_func(g, y_pred, y)

    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()

    if lr_scheduler:
        lr_scheduler.step()

    return (loss.item(), reg.item())


def validate_epoch(
    model: nn.Module,
    metric_func: nn.Module,
    valid_loader: MIODataLoader,
    device: str,
):
    model.eval()
    metric_val = []
    for _, data in enumerate(valid_loader):
        with torch.no_grad():
            g, u_p, g_u = data
            g, g_u, u_p = g.to(device), g_u.to(device), u_p.to(device)

            out = model(g, u_p, g_u)

            y_pred, y = out.squeeze(), g.ndata["y"].squeeze()
            _, _, metric = metric_func(g, y_pred, y)

            metric_val.append(metric)
    return dict(metric=np.mean(metric_val, axis=0))
