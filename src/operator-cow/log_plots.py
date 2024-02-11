import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb

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
    axs.legend()
    return plt
