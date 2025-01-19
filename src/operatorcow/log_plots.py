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
    fig, axs = plt.subplots(2, 2, figsize=(7.48/2, 9.45/2), dpi=300)
    #plt.xticks(fontsize=30)
    #plt.xticks(fontsize=8)
    #plt.yticks(fontsize=30)
    #fig.suptitle(f"Predictions vs Ground truth for {artery}")
    
    axs[0, 0].plot(unit_to_mmHg(ground_truth[:100, 0]), label="Ground truth", color="black", linewidth=1)
    axs[0, 0].plot(unit_to_mmHg(predictions[:100, 0]), label="Predicted", color="red", linestyle="--", linewidth=1)
    axs[0, 0].set_title("Pressure", fontsize=7)
    axs[0, 0].set_ylabel("mmHg", fontsize=7)
    axs[0, 0].set_xlabel("Time step", fontsize=7)
    axs[0, 0].tick_params(axis="x", labelsize=7)
    axs[0, 0].tick_params(axis="y", labelsize=7)
    plt.setp(axs[0,0].get_xticklabels(), fontsize=7,)
    plt.setp(axs[0,0].get_yticklabels(), fontsize=7,)
    # set legend size


    axs[0, 0].legend(fontsize=5)
    axs[0, 0].grid()

    axs[0, 1].plot(ground_truth[:100, 1], label="Ground truth", color="black",  linewidth=1)
    axs[0, 1].plot(predictions[:100, 1], label="Predicted", color="red", linestyle="--", linewidth=1)
   
    axs[0, 1].set_title("Area", fontsize=7)
    axs[0, 1].set_ylabel(r"$cm^2$", fontsize=7)
    axs[0, 1].set_xlabel("Time step", fontsize=7)
    axs[0, 1].tick_params(axis="x", labelsize=7)
    axs[0, 1].tick_params(axis="y", labelsize=7)
    plt.setp(axs[0,1].get_xticklabels(), fontsize=7,)
    plt.setp(axs[0,1].get_yticklabels(), fontsize=7,)
    org_lim = axs[0, 1].get_ylim()
    axs[0, 1].set_ylim([org_lim[0] - org_lim[0] * 0.10, org_lim[1] + org_lim[1] * 0.10])
    axs[0, 1].legend(fontsize=5)
    axs[0, 1].grid()
    axs[1, 0].plot(ground_truth[:100, 2], label="Ground truth", color="black", linewidth=1)
    axs[1, 0].plot(predictions[:100, 2], label="Predicted", color="red", linestyle="--", linewidth=1)
    
    axs[1, 0].set_title("Velocity", fontsize=7)
    axs[1, 0].set_ylabel(r"$ cm \cdot s^{-1} $", fontsize=7)
    axs[1, 0].set_xlabel("Time step", fontsize=7)
    axs[1, 0].tick_params(axis="x", labelsize=7)
    axs[1, 0].tick_params(axis="y", labelsize=7)
    plt.setp(axs[1,0].get_xticklabels(), fontsize=7,)
    plt.setp(axs[1,0].get_yticklabels(), fontsize=7,)
    axs[1, 0].legend(fontsize=5)
    axs[1, 0].grid()
    axs[1, 1].plot(ground_truth[:100, 2] * ground_truth[:100, 1], label="Ground truth", color="black", linewidth=1)
    axs[1, 1].plot(predictions[:100, 2] * predictions[:100, 1], label="Predicted", color="red", linestyle="--", linewidth=1)
    
    axs[1, 1].set_title("Flow", fontsize=7)
    axs[1, 1].set_ylabel(r"$cm^3 \cdot s^{-1}$", fontsize=7)
    axs[1, 1].set_xlabel("Time step", fontsize=7)
    axs[1, 1].tick_params(axis="x", labelsize=7)
    axs[1, 1].tick_params(axis="y", labelsize=7)
    plt.setp(axs[1,1].get_xticklabels(), fontsize=7,)
    plt.setp(axs[1,1].get_yticklabels(), fontsize=7,)
    axs[1, 1].legend(fontsize=5)
    axs[1, 1].grid()

    plt.tight_layout()
    plt.savefig(os.path.join(path, f"{artery}_FINAL.png"))
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
    # add artery mapping
    artery_mapping = {
        "ICA_1": "Internal carotid I",
        "BA": "Basilar",
        "MCA": "MCA",
        "ACA_A1": "ACA A1",
        "ACA_A2": "ACA A2",
        "PCA_P1": "PCA P1",
        "PCA_P2": "PCA P2",
        "PCOA": "PCoA",
        "ACOA": "ACoA",
        "ICA_2": "Internal carotid II",
    }

    fig, axs = plt.subplots(2, 5, figsize=(30, 30))
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
        z = torch.randn(512, model.get_latent_dim()).to(device) # sample true datase
        synthetic = model.decode(z, condition)
        synthetic = synthetic.reshape(512, 100, 1)
        synthetic = normalizer_y.transform(synthetic, inverse=True)
        synthetic = synthetic.detach().cpu().numpy()
        for i in range(synthetic.shape[0] - 1):
            axs[idx].plot(synthetic[i, :, 0], color="gray", alpha=0.2, linewidth=2)
        axs[idx].plot(synthetic[-1, :, 0], color="black", alpha=1, linewidth=2)
        axs[idx].set_title(artery)
        axs[idx].tick_params(axis="x", labelsize=30)
        axs[idx].tick_params(axis="y", labelsize=30)
        axs[idx].set_xlabel("Time step", fontsize=30)
        axs[idx].set_ylabel(r"$cm \cdot s^{-1}$", fontsize=30)
        
        axs[idx].grid()

    plt.tight_layout()
    plt.savefig(os.path.join(path, "VANO_samples_paper_new.png"))


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


def VANO_paper_full(
    model: torch.nn.Module,
    device: str,
    normalizer_up: torch.nn.Module,
    normalizer_y: torch.nn.Module,
    save_path: str,
    test_loader: torch.utils.data.DataLoader,
    ):
    """
    Function for creating plots and computing statistics from trainde vano model
    """
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
    artery_mapping = {
        "ICA_1": "Internal carotid I",
        "BA": "Basilar",
        "MCA": "MCA",
        "ACA_A1": "ACA A1",
        "ACA_A2": "ACA A2",
        "PCA_P1": "PCA P1",
        "PCA_P2": "PCA P2",
        "PCOA": "PCoA",
        "ACOA": "ACoA",
        "ICA_2": "Internal carotid II",
    }
    # loader outputs graph and condition
    # need to classify samples by artery and get 512 of each
    # then encode decode and plot samples
    # compute statsitics of samples from loader and then rfom vano

    # create dictionary for storing results artery wise for each variable
    results = {artery: {var: [] for var in ["true", "pred"]} for artery in arteries}
    for data in test_loader:
        if np.random.rand() > 0.5:
            continue
        with torch.no_grad():
            u_bc, condition = data
            u_bc = u_bc.ndata["y"].squeeze().reshape(condition.shape[0], -1).to(device)
            condition = condition.to(device)
            # get artery
            mean, log_var, z, out = model(u_bc, condition)

            z_samples = mean.unsqueeze(0) + torch.exp(log_var/2).unsqueeze(0) * torch.rand(z.shape, device=z.device)
            # loop over samples plot and compute statistics
            z_samples = z_samples.view(-1, z_samples.shape[-1])
            condition = condition.view(-1, condition.shape[-1])
            out = model.decode(z_samples, condition)
            #print(f"out shape: {out.shape}")
            #print(f"condition shape: {condition.shape}")
            out = normalizer_y.transform(out, inverse=True)
            u_bc = normalizer_y.transform(u_bc, inverse=True)
            #print(f"out shape: {out.shape}")
            #print(f"u_bc shape: {u_bc.shape}")
            
            for i in range(condition.shape[0]):
                artery = decode_artery(condition[i].cpu().numpy())
                true = u_bc[i].reshape(1, -1)
                pred = out[i].reshape(1, -1)
                if len(results[artery]["true"]) < 513:
                    results[artery]["true"].append(true)
                    results[artery]["pred"].append(pred)


    # plot samples and compute statistics 
    statistics_true = {}
    statistics_pred = {}
    fig_1, axs_1 = plt.subplots(2, 5, figsize=(7.48, 9.45/3), dpi=300)
    axs_1 = axs_1.flatten()
    fig_2, axs_2 = plt.subplots(2, 5, figsize=(7.48, 9.45/3), dpi=300)
    axs_2 = axs_2.flatten()
    for idx, artery in enumerate(arteries):
        print(artery)
        true = torch.cat(results[artery]["true"], dim=0)
        pred = torch.cat(results[artery]["pred"], dim=0)
        _, _, min, max, min_std, max_std = get_min_max(true)
        mfv, pi = compute_statistics(true)
        pi_std = torch.std(pi)
        mfv_std = torch.std(mfv)
        pi = torch.mean(pi)
        mfv = torch.mean(mfv)
        
        statistics_true[artery] = {
            "min": min,
            "max": max,
            "min_std": min_std,
            "max_std": max_std,
            "mfv": mfv,
            "mfv_std": mfv_std,
            "pi": pi,
            "pi_std": pi_std,
        }
        _, _, min, max, min_std, max_std = get_min_max(pred)
        mfv, pi = compute_statistics(pred)
        pi_std = torch.std(pi)
        mfv_std = torch.std(mfv)
        pi = torch.mean(pi)
        mfv = torch.mean(mfv)
        statistics_pred[artery] = {
            "min": min,
            "max": max,
            "min_std": min_std,
            "max_std": max_std,
            "mfv": mfv,
            "mfv_std": mfv_std,
            "pi": pi,
            "pi_std": pi_std,
        }

        # plot samples
        for i in range(len(results[artery]["true"]) - 1):
            #print(results[artery]["true"][i][0].shape)
            #import sys
            #sys.exit()
            axs_1[idx].plot(results[artery]["true"][i][0].cpu().numpy(), color="gray", alpha=0.2, linewidth=0.7)
            axs_2[idx].plot(results[artery]["pred"][i][0].cpu().numpy(), color="gray", alpha=0.2, linewidth=0.7)
        
        axs_1[idx].plot(results[artery]["true"][-1][0].cpu().numpy(), color="black", alpha=1, linewidth=0.7)
        axs_1[idx].set_title(artery_mapping[artery], fontsize=7)
        axs_1[idx].set_xlabel("Time step", fontsize=7)
        axs_1[idx].set_ylabel(r"$cm \cdot s^{-1}$", fontsize=7)
        axs_1[idx].tick_params(axis="x", labelsize=7)
        axs_1[idx].tick_params(axis="y", labelsize=7)
        axs_1[idx].grid()
        plt.setp(axs_1[idx].get_xticklabels(), fontsize=7,)
        plt.setp(axs_1[idx].get_yticklabels(), fontsize=7,)

        axs_2[idx].plot(results[artery]["pred"][-1][0].cpu().numpy(), color="black", alpha=1, linewidth=0.7)
        axs_2[idx].set_title(artery_mapping[artery], fontsize=7)
        axs_2[idx].set_xlabel("Time step", fontsize=7)
        axs_2[idx].set_ylabel(r"$cm \cdot s^{-1}$", fontsize=7)
        axs_2[idx].tick_params(axis="x", labelsize=7)
        axs_2[idx].tick_params(axis="y", labelsize=7)
        axs_2[idx].grid()
        plt.setp(axs_2[idx].get_xticklabels(), fontsize=7,)
        plt.setp(axs_2[idx].get_yticklabels(), fontsize=7,)
    fig_1.tight_layout()
    fig_2.tight_layout()
    fig_1.savefig(os.path.join(save_path, "VANO_samples_true_paper_test.png"))
    fig_2.savefig(os.path.join(save_path, "VANO_samples_pred_paper_test.png"))

    

            
    
    
    
    # save statistcs to csv file
    df_true = pd.DataFrame(statistics_true)
    df_true.to_csv(f"{save_path}/statistics_true_VANO_new.csv")
    df_pred = pd.DataFrame(statistics_pred)
    df_pred.to_csv(f"{save_path}/statistics_pred_VANO_new.csv")


        # plot samples
        