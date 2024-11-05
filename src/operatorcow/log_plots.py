import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import wandb
from utils.GANO_utils import compute_statistics, get_min_max

# Functions for logging predicitions and ground truth
# for vizualization in wandb plotting true vs predicted


def decode_artery(artery: np.ndarray) -> str:
    """
    Function decodes artery one-hot vector to artery name
    """
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
    index = np.argmax(artery)
    return arteries[index]


def encode_artery(artery: str) -> np.ndarray:
    """
    Function encodes artery type as one-hot vector

    Parameters
    ----------
    artery : str
        Artery type must be one of the following:
        "VA", "ICA_1", "BA", "MCA", "ACA_A1", "ACA_A2",
        "PCA_P1", "PCA_P2", "PCOA", "ACOA", "ICA_2"
    """
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
    out = np.zeros(len(arteries))
    index = arteries.index(artery)
    out[index] = 1
    return out


def plot_predictions(
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    artery: str,
) -> None:
    """
    Function for creating plot for comparison preditions and ground truth
    and logiing it into wandb

    ground truth and predictions are in form:
    withe entries: [
        [p(x0, t0), A(x0, t0), u(x0, t0)],
        [p(x0, t1), A(x0, t1), u(x0, t1)],
        ...
        [p(xn, t0), A(xn, t0), u(xn, t0)],
        [p(xn, t1), A(xn, t1), u(xn, t1)],
    ]
    """
    print("Plotting")
    fig, axs = plt.subplots(3, 1, figsize=(10, 10))
    fig.suptitle(f"Predictions vs Ground truth for {artery} at inlet")
    axs[0].plot(predictions[:100, 0], label="Predicted")
    axs[0].plot(ground_truth[:100, 0], label="Ground truth")
    axs[0].set_title("Pressure")
    axs[0].legend()

    axs[1].plot(predictions[:100, 1], label="Predicted")
    axs[1].plot(ground_truth[:100, 1], label="Ground truth")
    axs[1].set_title("Area")
    axs[1].legend()

    axs[2].plot(predictions[:100, 2], label="Predicted")
    axs[2].plot(ground_truth[:100, 2], label="Ground truth")
    axs[2].set_title("Velocity")
    axs[2].legend()

    wandb.log({f"{artery} inlet": plt})
    plt.close()

    # plot at outlet
    fig, axs = plt.subplots(3, 1, figsize=(10, 10))
    fig.suptitle(f"Predictions vs Ground truth for {artery} at outlet")
    axs[0].plot(predictions[-100:, 0], label="Predicted")
    axs[0].plot(ground_truth[-100:, 0], label="Ground truth")
    axs[0].set_title("Pressure")
    axs[0].legend()

    axs[1].plot(predictions[-100:, 1], label="Predicted")
    axs[1].plot(ground_truth[-100:, 1], label="Ground truth")
    axs[1].set_title("Area")
    axs[1].legend()

    axs[2].plot(predictions[-100:, 2], label="Predicted")
    axs[2].plot(ground_truth[-100:, 2], label="Ground truth")
    axs[2].set_title("Velocity")
    axs[2].legend()

    wandb.log({f"{artery} outlet": plt})
    plt.close()


def plot_generated(velocity: torch.Tensor) -> plt:
    """
    Function for plotting generated samples of velocity
    all batch on single plot
    """
    fig, axs = plt.subplots(1, 1, figsize=(10, 10))
    fig.suptitle("Generated samples of velocity")
    for i in range(velocity.shape[0]):
        axs.plot(velocity[i, :, 0].detach().cpu().numpy())
    # axs.plot(velocity[0, :, 0].detach().cpu().numpy(), label="Generated")
    axs.set_title("Velocity")
    # axs.legend()
    return plt


def plot_AE(pred: torch.Tensor, target: torch.Tensor, type: str) -> plt:
    """
    Function for plotting and logging AE reconsturctions
    and targets
    """
    if type not in ["target", "reconstruction", "comparison"]:
        raise ValueError(
            "type must be one of the following: 'target', 'reconstruction', 'comparison'"
        )

    if type == "comparison":
        fig, axs = plt.subplots(1, 1, figsize=(10, 10))
        fig.suptitle("Reconstruction vs Target")
        axs.plot(pred[0, :, 0].detach().cpu().numpy(), label="Reconstruction")
        axs.plot(target[0, :, 0].detach().cpu().numpy(), label="Target")
        axs.set_title("velocity")
        axs.legend()
        return plt
    elif type == "target":
        fig, axs = plt.subplots(1, 1, figsize=(10, 10))
        fig.suptitle("Target")
        axs.plot(target[0, :, 0].detach().cpu().numpy(), label="Target")
        axs.set_title("velocity")
        axs.legend()
        return plt
    else:
        fig, axs = plt.subplots(1, 1, figsize=(10, 10))
        fig.suptitle("Reconstruction")
        axs.plot(pred[0, :, 0].detach().cpu().numpy(), label="Reconstruction")
        axs.set_title("velocity")
        axs.legend()
        return plt


def unit_to_mmHg(p: np.ndarray) -> np.ndarray:
    """
    Function for converting pressure from units to mmHg
    """
    return 76 / 101325 * p


def plot_pred_paper(
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    artery: str,
    path: str,
) -> None:
    """
    Function for creating plot for comparison preditions and ground truth
    and logiing it into wandb

    ground truth and predictions are in form:
    withe entries: [
        [p(x0, t0), A(x0, t0), u(x0, t0)],
        [p(x0, t1), A(x0, t1), u(x0, t1)],
        ...
        [p(xn, t0), A(xn, t0), u(xn, t0)],
        [p(xn, t1), A(xn, t1), u(xn, t1)],
    ]
    """
    print("Plotting")
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    fig.suptitle(f"Predictions vs Ground truth for {artery}")
    
    axs[0, 0].plot(unit_to_mmHg(ground_truth[:100, 0]), label="Ground truth", color="black")
    axs[0, 0].plot(unit_to_mmHg(predictions[:100, 0]), label="Predicted", color="red", linestyle="--")
    axs[0, 0].set_title("Pressure")
    axs[0, 0].set_ylabel("mmHg")
    axs[0, 0].set_xlabel("Time step")
    axs[0, 0].legend()
    axs[0, 0].grid()

    axs[0, 1].plot(ground_truth[:100, 1], label="Ground truth", color="black")
    axs[0, 1].plot(predictions[:100, 1], label="Predicted", color="red", linestyle="--")
   
    axs[0, 1].set_title("Area")
    axs[0, 1].set_ylabel("cm^2")
    axs[0, 1].set_xlabel("Time step")
    org_lim = axs[0, 1].get_ylim()
    axs[0, 1].set_ylim([org_lim[0] - org_lim[0] * 0.10, org_lim[1] + org_lim[1] * 0.10])
    axs[0, 1].legend()
    axs[0, 1].grid()
    axs[1, 0].plot(ground_truth[:100, 2], label="Ground truth", color="black")
    axs[1, 0].plot(predictions[:100, 2], label="Predicted", color="red", linestyle="--")
    
    axs[1, 0].set_title("Velocity")
    axs[1, 0].set_ylabel("cm/s")
    axs[1, 0].set_xlabel("Time step")
    axs[1, 0].legend()
    axs[1, 0].grid()
    axs[1, 1].plot(ground_truth[:100, 2] * ground_truth[:100, 1], label="Ground truth", color="black")
    axs[1, 1].plot(predictions[:100, 2] * predictions[:100, 1], label="Predicted", color="red", linestyle="--")
    
    axs[1, 1].set_title("Flow")
    axs[1, 1].set_ylabel("cm^3/s")
    axs[1, 1].set_xlabel("Time step")
    axs[1, 1].legend()
    axs[1, 1].grid()

    plt.savefig(os.path.join(path, f"{artery}.png"))
    plt.close()


def create_plot_VANO_paper(
    model: torch.nn.Module,
    device: str,
    normalizer_up: torch.nn.Module,
    normalizer_y: torch.nn.Module,
    path: str,
):
    model.eval()
    arteries = [
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
    fig, axs = plt.subplots(2, 5, figsize=(15, 6))
    axs = axs.flatten()
    for idx, artery in enumerate(arteries):
        print(artery)
        artery_one_hot = encode_artery(artery)
        condition = (
            torch.from_numpy(artery_one_hot)
            .float()
            .to(device)
            .unsqueeze(0)
            .repeat(512, 1)
        )
        condition = normalizer_up.transform(condition, inverse=False)
        z = torch.randn(512, model.get_latent_dim()).to(device)
        synthetic = model.decode(z, condition)
        synthetic = synthetic.reshape(512, 100, 1)
        synthetic = normalizer_y.transform(synthetic, inverse=True)
        synthetic = synthetic.detach().cpu().numpy()
        for i in range(synthetic.shape[0] - 1):
            axs[idx].plot(synthetic[i, :, 0], color="gray", alpha=0.2, linewidth=1)
        axs[idx].plot(synthetic[-1, :, 0], color="black", alpha=1, linewidth=1)
        axs[idx].set_title(artery)
        axs[idx].set_xlabel("Time step")
        axs[idx].set_ylabel("cm/s")
        axs[idx].grid()

    plt.tight_layout()
    plt.savefig(os.path.join(path, "VANO_samples_paper.png"))


def compute_statistics_VANO_paper(
    model: torch.nn.Module,
    device: str,
    normalizer_up: torch.nn.Module,
    normalizer_y: torch.nn.Module,
    save_path: str,
):
    model.eval()
    arteries = [
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
    statistics = {}
    for idx, artery in enumerate(arteries):
        print(artery)
        artery_one_hot = encode_artery(artery)
        condition = (
            torch.from_numpy(artery_one_hot)
            .float()
            .to(device)
            .unsqueeze(0)
            .repeat(512, 1)
        )
        condition = normalizer_up.transform(condition, inverse=False)
        z = torch.randn(512, model.get_latent_dim()).to(device)
        synthetic = model.decode(z, condition)
        synthetic = synthetic.reshape(512, 100, 1)
        synthetic = normalizer_y.transform(synthetic, inverse=True)

        _, _, min, max, min_std, max_std = get_min_max(synthetic)
        mfv, pi = compute_statistics(synthetic)
        mfv = torch.mean(mfv)
        mfv_std = torch.std(mfv)
        pi = torch.mean(pi)
        pi_std = torch.std(pi)
        statistics[artery] = {
            "min": min,
            "max": max,
            "min_std": min_std,
            "max_std": max_std,
            "mfv": mfv,
            "mfv_std": mfv_std,
            "pi": pi,
            "pi_std": pi_std,
        }
    # save statistcs to csv file
    df = pd.DataFrame(statistics)
    df.to_csv(f"{save_path}/statistics_VANO.csv")

    return statistics
