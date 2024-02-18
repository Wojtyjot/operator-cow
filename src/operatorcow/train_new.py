import os
import pickle
import sys

import dgl
import numpy as np
import torch
import torch.nn as nn
import wandb
from data_utils import COWDataset, MIODataLoader, calculate_gradient_penalty
from grf import GaussianRF
from log_plots import (
    decode_artery,
    encode_artery,
    plot_AE,
    plot_generated,
    plot_predictions,
)
from models.optimizer import Adam
from torch.optim.lr_scheduler import OneCycleLR
from utils.GANO_utils import compute_statistics
from utils.utils import MultipleTensors, get_num_params


def train(
    model: nn.Module,
    loss_func: nn.Module,
    metric_func: nn.Module,
    train_loader: MIODataLoader,
    val_loader: MIODataLoader,
    optimizer,
    lr_scheduler,
    epochs: int,
    device: str,
    grad_clip: float,
    normalizer_up,
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
        model.train()
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
            normalizer_up=normalizer_up,
        )
        wandb.log({"val_L2_loss": val_result["metric"].mean()})

        val_metric = val_result["metric"].sum()
        loss_val.append(val_metric)

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
    optimizer,
    lr_scheduler,
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
    normalizer_up,
):
    model.eval()
    metric_val = []
    plotted_vessels = []  # 10 arteries thetas [0:10]
    for _, data in enumerate(valid_loader):
        with torch.no_grad():
            g, u_p, g_u = data
            g, g_u, u_p = g.to(device), g_u.to(device), u_p.to(device)

            out = model(g, u_p, g_u)

            y_pred, y = out.squeeze(), g.ndata["y"].squeeze()
            _, _, metric = metric_func(g, y_pred, y)

            metric_val.append(metric)
            # TRZEBA ZRENORMALIZOWAĆ !!!!!!!!!
            if len(plotted_vessels) < 10:

                u_p = normalizer_up.transform(u_p, inverse=True)
                artery = decode_artery(u_p[0][:10].cpu().numpy())  # tu nie 11?

                if artery not in plotted_vessels:
                    plot_predictions(y_pred.cpu().numpy(), y.cpu().numpy(), artery)
                    plotted_vessels.append(artery)

    return dict(metric=np.mean(metric_val, axis=0))


# trzeba zobaczyc cz lr_scheduler w gano
def train_GANO(
    Generator: nn.Module,
    Discriminator: nn.Module,
    # loss_func: nn.Module,
    # metric_func: nn.Module,
    train_loader: MIODataLoader,
    # val_loader: MIODataLoader,
    optimizer_discriminator,
    optimizer_generator,
    lr_scheduler_generator,
    lr_scheduler_discriminator,
    epochs: int,
    device: str,
    grad_clip: float,
    normalizer_up,
    normalizer_x,
    normalizer_y,
    grf: GaussianRF,
    lambda_grad: float = 10.0,
    start_epoch: int = 0,
    print_freq: int = 20,
    model_save_path: str = "../../data/checkpoints/",
    Generator_name: str = "Generator_GANO.pt",
    Discriminator_name: str = "Discriminator_GANO.pt",
    result_name: str = "results_GANO.pt",
):

    it = 0
    best_val_metric = np.inf
    n_critic = 20

    for epoch in range(epochs):
        Generator.train()
        Discriminator.train()
        torch.cuda.empty_cache()

        for j, batch in enumerate(train_loader):
            # train discriminator
            W_loss, D_loss, grad_loss = train_batch_GANO_D(
                Generator=Generator,
                Discriminator=Discriminator,
                data=batch,
                optimizer_discriminator=optimizer_discriminator,
                device=device,
                grad_clip=grad_clip,
                grf=grf,
                lambda_grad=lambda_grad,
                lr_scheduler_discriminator=lr_scheduler_discriminator,
            )
            if (j + 1) % n_critic == 0:
                # train generator
                G_loss = train_batch_GANO_G(
                    Generator=Generator,
                    Discriminator=Discriminator,
                    data=batch,
                    optimizer_generator=optimizer_generator,
                    device=device,
                    grad_clip=grad_clip,
                    grf=grf,
                    lr_scheduler_generator=lr_scheduler_generator,
                )
                wandb.log({"G_loss": G_loss})
            wandb.log(
                {
                    "W_loss": W_loss,
                    "D_loss": D_loss,
                    "grad_loss": grad_loss,
                    "lr_D": optimizer_discriminator.param_groups[0]["lr"],
                    "lr_G": optimizer_generator.param_groups[0]["lr"],
                }
            )

            it += 1
            log = f"epoch: [{epoch +1}/{epochs}]"
            log += f"W_loss: {W_loss:.4f}"
            log += f"D_loss: {D_loss:.4f}"
            log += f"grad_loss: {grad_loss:.4f}"

            if (j + 1) % n_critic == 0:
                log += f"G_loss: {G_loss:.4f}"

            if it % print_freq == 0:
                print(log)

        # validate epoch to bedzie policzenie statystyk dla generatora
        # oraz zrobienie wykresów dla 10 naczyń
        validate_epoch_GANO(
            Generator=Generator,
            device=device,
            normalizer_up=normalizer_up,
            normalizer_x=normalizer_x,
            normalizer_y=normalizer_y,
            grf=grf,
        )

        # save models:
        # save Generator
        torch.save(
            Generator.state_dict(),
            os.path.join(model_save_path, Generator_name),
        )
        # save Discriminator
        torch.save(
            Discriminator.state_dict(),
            os.path.join(model_save_path, Discriminator_name),
        )

    return None


def train_batch_GANO_G(
    Generator: nn.Module,
    Discriminator: nn.Module,
    # loss_func: nn.Module,
    data: list,
    optimizer_generator,
    lr_scheduler_generator,
    device: str,
    grad_clip: float,
    grf: GaussianRF,
):
    optimizer_generator.zero_grad()
    g, u_p = data

    # sample from gaussian random field num of samples = batch_size
    N = u_p.shape[0]
    z = grf.sample(N, mul=1).unsqueeze(-1)
    t = torch.linspace(0, 1, 100).unsqueeze(-1).repeat(N, 1, 1)
    z = torch.cat((t, z), dim=-1)
    z = MultipleTensors([z])

    g, u_p, z = g.to(device), u_p.to(device), z.to(device)
    t = t.to(device)

    synthetic = Generator(g, u_p, z)
    synthetic = synthetic.reshape(N, 100, 1)
    synthetic = torch.cat((t, synthetic), dim=-1)
    synthetic = MultipleTensors([synthetic])

    loss = -torch.mean(Discriminator(g, u_p, synthetic))

    loss.backward()
    nn.utils.clip_grad_norm_(Generator.parameters(), grad_clip)
    optimizer_generator.step()

    if lr_scheduler_generator:
        lr_scheduler_generator.step()

    return loss.item()


def train_batch_GANO_D(
    Generator: nn.Module,
    Discriminator: nn.Module,
    # loss_func: nn.Module,
    data: list,
    optimizer_discriminator,
    lr_scheduler_discriminator,
    device: str,
    grad_clip: float,
    grf: GaussianRF,
    lambda_grad: float = 10.0,
):
    optimizer_discriminator.zero_grad()
    g, u_p = data

    # sample from gaussian random field num of samples = batch_size
    N = u_p.shape[0]
    z = grf.sample(N, mul=1).unsqueeze(-1)
    t = torch.linspace(0, 1, 100).unsqueeze(-1).repeat(N, 1, 1)
    z = torch.cat((t, z), dim=-1)
    z = MultipleTensors([z])
    # real already with t concatenated
    # need to cat z with t

    g, u_p, z = g.to(device), u_p.to(device), z.to(device)
    t = t.to(device)

    synthetic = Generator(g, u_p, z)
    synthetic = synthetic.reshape(N, 100, 1)
    # need to pass it to discriminator and calculate loss
    synthetic = torch.cat((t, synthetic), dim=-1)

    real = g.ndata["y"]
    real = real.reshape(N, 100, 1)
    real = torch.cat((t, real), dim=-1)
    real = MultipleTensors([real])
    synthetic = MultipleTensors([synthetic.detach()])

    ### passs to discriminator
    W_loss = -torch.mean(Discriminator(g, u_p, real)) + torch.mean(
        Discriminator(g, u_p, synthetic)
    )

    # y_pred, y = out.squeeze(), g.ndata["y"].squeeze()
    gradient_penalty = calculate_gradient_penalty(
        Discriminator=Discriminator,
        real_function=real[0],
        fake_function=synthetic[0],
        device=device,
        g=g,
        u_p=u_p,
        t=t,
    )

    loss = W_loss + lambda_grad * gradient_penalty

    # loss, reg, _ = loss_func(g, y_pred, y)

    loss.backward()
    nn.utils.clip_grad_norm_(Discriminator.parameters(), grad_clip)
    optimizer_discriminator.step()

    if lr_scheduler_discriminator:
        lr_scheduler_discriminator.step()

    return (W_loss.item(), loss.item(), gradient_penalty.item())


def validate_epoch_GANO(
    Generator: nn.Module,
    device: str,
    normalizer_up,
    normalizer_x,  # musza byc juz na device
    normalizer_y,
    grf: GaussianRF,
    # num_samples: int = 80,
):
    """
    Function generates samples from generator
    and computes staticsics + save plots and tables
    to wandb
    """
    Generator.eval()
    arteries = [
        "VA",
        "ICA_1",
        "BA",
        "MCA",
        "ACA_A1",
        "ACA_A2",
        "PCA_P1",
        "PCA_P2",
        "PCOA",
        "ACOA",
        "ICA_2",
    ]
    tbl = wandb.Table(
        columns=["Artery", "MFV_mean", "MFV_std", "PI_mean", "PI_std", "Sample"]
    )
    for artery in arteries:
        artery_one_hot = encode_artery(artery)
        u_p = (
            torch.from_numpy(artery_one_hot)
            .float()
            .to(device)
            .unsqueeze(0)
            .repeat(8, 1)
        )
        # normalize
        u_p = normalizer_up.transform(u_p, inverse=False)
        g = dgl.DGLGraph()
        g.add_nodes(100)
        g.ndata["x"] = normalizer_x.transform(
            torch.linspace(0, 1, 100).reshape(-1, 1).to(device), inverse=False
        )
        gs = dgl.batch([g for _ in range(8)])
        MFV, PI = None, None
        for i in range(10):
            z = grf.sample(8, mul=1).unsqueeze(-1)
            t = torch.linspace(0, 1, 100).unsqueeze(-1).repeat(8, 1, 1)
            z = torch.cat((t, z), dim=-1)
            z = MultipleTensors([z])
            gs, z, u_p = gs.to(device), z.to(device), u_p.to(device)
            synthetic = Generator(gs, u_p, z)
            synthetic = synthetic.reshape(8, 100, 1)  # (8, 100, 1)
            # Need to denormalize and compute statistics
            synthetic = normalizer_y.transform(synthetic, inverse=True)
            if MFV is None:
                MFV, PI = compute_statistics(synthetic)
            else:
                MFV_new, PI_new = compute_statistics(synthetic)
                MFV = torch.cat((MFV, MFV_new), dim=0)
                PI = torch.cat((PI, PI_new), dim=0)
        MFV_mean, MFV_std = torch.mean(MFV, dim=0), torch.std(MFV, dim=0)
        PI_mean, PI_std = torch.mean(PI, dim=0), torch.std(PI, dim=0)
        tbl.add_data(
            artery,
            MFV_mean.item(),
            MFV_std.item(),
            PI_mean.item(),
            PI_std.item(),
            wandb.Image(plot_generated(synthetic)),
        )
    wandb.log({"Generated_distribution_statistics": tbl})


def train_AE(
    model: nn.Module,
    loss_func: nn.Module,
    metric_func: nn.Module,
    train_loader: MIODataLoader,
    val_loader: MIODataLoader,
    optimizer,
    lr_scheduler,
    epochs: int,
    device: str,
    grad_clip: float,
    normalizer_y,
    start_epoch: int = 0,
    print_freq: int = 20,
    model_save_path: str = "../../data/checkpoints/",
    model_name: str = "model.pt",
    result_name: str = "results.pt",
):

    it = 0
    best_val_metric = np.inf

    for epoch in range(epochs):
        model.train()
        torch.cuda.empty_cache()

        for batch in train_loader:
            loss = train_batch_AE(
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
                    "MSEloss": loss,
                    "lr": optimizer.param_groups[0]["lr"],
                }
            )
            loss = np.array(loss)
            it += 1
            log = f"epoch: [{epoch +1}/{epochs}]"
            log += f"loss: {loss:.4f}"

            if it % print_freq == 0:
                print(log)

        val_result = validate_epoch_AE(
            model=model,
            metric_func=metric_func,
            valid_loader=val_loader,
            device=device,
            normalizer_y=normalizer_y,
        )
        wandb.log({"val_MSE_loss": val_result["metric"].mean()})

        val_metric = val_result["metric"].sum()

        if val_metric < best_val_metric:
            best_val_metric = val_metric
            best_val_epoch = epoch
            torch.save(
                model.state_dict(),
                os.path.join(model_save_path, model_name),
            )

        return None


def train_batch_AE(
    model: nn.Module,
    loss_func: nn.Module,
    data: list,
    optimizer,
    lr_scheduler,
    device: str,
    grad_clip: float,
):
    # func to encode g.ndata['y']
    optimizer.zero_grad()
    g, _ = data
    target = g.ndata["y"].squeeze()
    target = target.reshape(target.shape[0] // 100, 100)

    target = target.to(device)
    out = model(target)

    loss = loss_func(out, target)

    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()

    if lr_scheduler:
        lr_scheduler.step()

    return loss.item()


def validate_epoch_AE(
    model: nn.Module,
    metric_func: nn.Module,
    valid_loader: MIODataLoader,
    device: str,
    normalizer_y,
):
    tbl = wandb.Table(columns=["Target", "MSE_loss", "Predicted", "Comparison"])
    model.eval()
    metric_val = []
    for _, data in enumerate(valid_loader):
        with torch.no_grad():
            g, _ = data
            target = g.ndata["y"].squeeze()
            target = target.to(device)

            out = model(target)
            metric = metric_func(out, target)
            metric_val.append(metric.detach().cpu().numpy())

    g, _ = data
    target = g.ndata["y"].squeeze()
    target = target.to(device)
    out = model(target)
    target = normalizer_y.transform(target, inverse=True)
    out = normalizer_y.transform(out, inverse=True)
    metric = metric_func(out, target)
    tbl.add_data(
        wandb.Image(plot_AE(out, target, type="target")),  # obczaić te funkcje
        wandb.Image(plot_AE(out, target, type="reconstruction")),
        metric.item(),
        wandb.Image(plot_AE(out, target, type="comparison")),
    )
    wandb.log({"AE_validation": tbl})

    return dict(metric=np.mean(metric_val, axis=0))
