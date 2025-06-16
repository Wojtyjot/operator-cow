import logging
from pathlib import Path

import hydra
import torch
import torch.nn as nn
import wandb
from data_utils import COWDataset, COWDataset_GANO, MIODataLoader, WeightedLpRelLoss, COWDataset_res
from log_plots import (
    compute_statistics_VANO_paper,
    create_plot_VANO_paper,
    decode_artery,
    plot_pred_paper,
    VANO_paper_full,
    unit_to_mmHg,
)
from models.cgpt import CGPT
from models.mmgpt import GNOT
from models.optimizer import AdamW
from models.VANO import VANO
from omegaconf import DictConfig, OmegaConf
from torch.optim.lr_scheduler import OneCycleLR
from train_new import train
from utils import seeding, utils
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

OmegaConf.register_new_resolver(
    "generate_random_seed", seeding.generate_random_seed, use_cache=True
)


@hydra.main(version_base=None, config_path="configs", config_name="Plots")
def main(config: DictConfig) -> None:
    if False:
        wandb.init(
            config=OmegaConf.to_container(config, resolve=True, throw_on_missing=True),
            project=config.wandb.project,
            tags=config.wandb.tags,
            anonymous=config.wandb.anonymous,
            mode=config.wandb.mode,
            dir=Path(config.wandb.dir).absolute(),
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load the data
    
    if False:
        dataset_train = COWDataset_GANO(config.data.path_train)
        dataset_test = COWDataset_GANO(config.data.path_test)
    elif True:

        #dataset_train = COWDataset(config.data.path_train)
        dataset_test = COWDataset(config.data.path_test)
    else:
        dataset_train = COWDataset_res(config.data.path_train)
        dataset_test = COWDataset_res(config.data.path_test)

    torch.manual_seed(config.seed)
    # save normalizer to renormalize data for plotting and evaluation
    # for vano only normalizer_u_bc = normalizer_y 
    normalizer_u_bc = dataset_test.y_normalizer.to(device)
   # print(f"normalizer_y mean: {normalizer_u_bc.mean}")
    #print(f"normalizer_y std: {normalizer_u_bc.std}")
    #print(f"normalizer_u_bc mean: {normalizer_u_bc}")
    #sys.exit()
    normalizer = dataset_test.y_normalizer.to(device)
    normalizer_up = dataset_test.up_normalizer.to(device)
    normalizer_x = dataset_test.x_normalizer.to(device)
    #print(f"x_norm mean: {normalizer_x.mean}")
    #print(f"x_norm std: {normalizer_x.std}")
    #sys.exit()
    


    

    #train_loader = MIODataLoader(
    #    dataset_train,
    #    batch_size=config.data.batch_size,
    #    shuffle=True,
    #    drop_last=False,
    #)
    test_loader = MIODataLoader(
        dataset_test,
        batch_size=1, # FOR VANO ONLY !!! or 512?
        shuffle=True,
        drop_last=False,
    )
    

    # Load model
    if config.model.name == "CGPT":

        model = CGPT(
            trunk_size=dataset_train.config["input_dim"]
            + dataset_train.config["theta_dim"],
            branch_sizes=dataset_train.config["branch_sizes"],
            output_size=dataset_train.config["output_dim"],
            space_dim=2,
            n_layers=config.model.n_layers,
            n_hidden=config.model.n_hidden,
            n_head=config.model.n_head,
            n_experts=config.model.n_experts,
            n_inner=config.model.n_inner,
            mlp_layers=config.model.mlp_layers,
            attn_type=config.model.attn_type,
            act=config.model.act,
            ffn_dropout=config.model.ffn_dropout,
            attn_dropout=config.model.attn_dropout,
            horiz_fourier_dim=config.model.horiz_fourier_dim,
        )

    elif config.model.name == "GNOT":

        model = GNOT(
            trunk_size=52,
            branch_sizes=[3,3],
            output_size=3,
            space_dim=2,
            n_layers=config.model.n_layers,
            n_hidden=config.model.n_hidden,
            n_head=config.model.n_head,
            n_experts=config.model.n_experts,
            n_inner=config.model.n_inner,
            mlp_layers=config.model.mlp_layers,
            attn_type=config.model.attn_type,
            act=config.model.act,
            ffn_dropout=config.model.ffn_dropout,
            attn_dropout=config.model.attn_dropout,
            horiz_fourier_dim=config.model.horiz_fourier_dim,
        )

    else:
        raise ValueError(f"Model {config.model.name} not recognized.")

    loss_func = WeightedLpRelLoss(
        p=2,
        component="all",
        normalizer=None,
    )
   

    metric_func = WeightedLpRelLoss(
        p=2,
        component="all",
        normalizer=normalizer,
    )
    VANO_model = VANO(
        layers_encoder=config.model.VANO.layers_encoder,
        layers_decoder=config.model.VANO.layers_decoder,
        latent_dim=config.model.VANO.latent_dim,
    )
    #VANO_model.load_state_dict(torch.load(config.model.VANO_weights_path))

    model.load_state_dict(torch.load(config.model.surrogate_weights_path))
    VANO_model.to(device)
    VANO_model.eval()
    if False:
        VANO_paper_full(
            model=VANO_model,
            device=device,
            normalizer_up=normalizer_up,
            normalizer_y=normalizer,
            save_path=config.plots.path,
            test_loader=test_loader,
        )
        
    else:

        model.eval()
        model.to(device)
        ploted_vessels = []
        arteries = [
            # "VA",
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
        vars = ["pressure", "velocity", "area"]
        # create dictionary for storing results artery wise for each variable
        results = {var: {artery: [] for artery in arteries} for var in vars}
        full = []
        for data in test_loader:
            with torch.no_grad():
                g, u_p, g_u = data
                g, g_u, u_p = g.to(device), g_u.to(device), u_p.to(device)
                # import time
                import sys

                # st = time.time()
                out = model(g, u_p, g_u)
                # print(time.time()-st)

                # sys.exit()
                out = normalizer.transform(out, inverse=True)
                y = normalizer.transform(g.ndata["y"], inverse=True)
                #print(f"out shape: {out.shape}")
                #print(f"y shape: {y.shape}")

                y_pred, y = out.squeeze(), y.squeeze()
                if len(ploted_vessels) < 11:
                    if len(ploted_vessels) == 0:
                        fig, axs = plt.subplots(2,5, figsize=(15,6))
                    u_p = normalizer_up.transform(u_p, inverse=True)
                    artery = decode_artery(u_p[0][:11].cpu().numpy())  # tu nie 11?
                    #print(axs[len(ploted_vessels)].shape)

                    if artery not in ploted_vessels:
                        #plot_pred_paper(
                        #    y_pred.cpu().numpy(),
                        #    y.cpu().numpy(),
                        #    artery,
                        #    config.plots.path,
                        #)
                        #axs[len(ploted_vessels)//5, len(ploted_vessels)%5].plot(y.cpu().numpy()[:100,0], label="Ground Truth", color="black")
                        if True:
                            axs[len(ploted_vessels)//5, len(ploted_vessels)%5].plot(unit_to_mmHg(y.cpu().numpy()[:100,0]), label="Ground Truth", color="black")
                            axs[len(ploted_vessels)//5, len(ploted_vessels)%5].plot(unit_to_mmHg(y_pred.cpu().numpy()[:100,0]), label="Predicted", color="red", linestyle="--")
                            axs[len(ploted_vessels)//5, len(ploted_vessels)%5].set_title(artery_mapping[artery], fontsize=12)
                            axs[len(ploted_vessels)//5, len(ploted_vessels)%5].tick_params(axis="x", labelsize=12)
                            axs[len(ploted_vessels)//5, len(ploted_vessels)%5].tick_params(axis="y", labelsize=12)
                            axs[len(ploted_vessels)//5, len(ploted_vessels)%5].set_xlabel("Time step", fontsize=12)
                            axs[len(ploted_vessels)//5, len(ploted_vessels)%5].set_ylabel("mmHg", fontsize=12)
                            axs[len(ploted_vessels)//5, len(ploted_vessels)%5].legend(fontsize=8)
                            axs[len(ploted_vessels)//5, len(ploted_vessels)%5].grid()
                        elif True:
                            axs[len(ploted_vessels)//5, len(ploted_vessels)%5].plot(y.cpu().numpy()[:100,1], label="Ground Truth", color="black")
                            axs[len(ploted_vessels)//5, len(ploted_vessels)%5].plot(y_pred.cpu().numpy()[:100,1], label="Predicted", color="red", linestyle="--")
                            axs[len(ploted_vessels)//5, len(ploted_vessels)%5].set_title(artery_mapping[artery], fontsize=12)
                            axs[len(ploted_vessels)//5, len(ploted_vessels)%5].tick_params(axis="x", labelsize=12)
                            axs[len(ploted_vessels)//5, len(ploted_vessels)%5].tick_params(axis="y", labelsize=12)
                            axs[len(ploted_vessels)//5, len(ploted_vessels)%5].set_xlabel("Time step", fontsize=12)
                            axs[len(ploted_vessels)//5, len(ploted_vessels)%5].set_ylabel(r"$cm^2$", fontsize=12)
                            axs[len(ploted_vessels)//5, len(ploted_vessels)%5].legend(fontsize=8)
                            axs[len(ploted_vessels)//5, len(ploted_vessels)%5].grid()

                        elif True:
                            axs[len(ploted_vessels)//5, len(ploted_vessels)%5].plot(y.cpu().numpy()[:100,2], label="Ground Truth", color="black")
                            axs[len(ploted_vessels)//5, len(ploted_vessels)%5].plot(y_pred.cpu().numpy()[:100,2], label="Predicted", color="red", linestyle="--")
                            axs[len(ploted_vessels)//5, len(ploted_vessels)%5].set_title(artery_mapping[artery], fontsize=12)
                            axs[len(ploted_vessels)//5, len(ploted_vessels)%5].tick_params(axis="x", labelsize=12)
                            axs[len(ploted_vessels)//5, len(ploted_vessels)%5].tick_params(axis="y", labelsize=12)
                            axs[len(ploted_vessels)//5, len(ploted_vessels)%5].set_xlabel("Time step", fontsize=12)
                            axs[len(ploted_vessels)//5, len(ploted_vessels)%5].set_ylabel(r"$ cm \cdot s^{-1} $", fontsize=12)
                            axs[len(ploted_vessels)//5, len(ploted_vessels)%5].legend(fontsize=8)
                            axs[len(ploted_vessels)//5, len(ploted_vessels)%5].grid()


                        ploted_vessels.append(artery)
                artery = decode_artery(u_p[0][:11].cpu().numpy())
                
                error_pressure, _, _ = loss_func(g, y_pred[:, 0], y[:, 0])
                error_area, _, _ = loss_func(g, y_pred[:, 1], y[:, 1])
                error_velocity, _, _ = loss_func(g, y_pred[:, 2], y[:, 2])
                results["pressure"][artery].append(error_pressure)
                results["area"][artery].append(error_area)
                results["velocity"][artery].append(error_velocity)
                error_full, _, _ = loss_func(g, y_pred, y)
                full.append(error_full)
                if len(ploted_vessels) == 10:
                    break
        plt.tight_layout()
        plt.savefig(f"{config.plots.path}/Pressure.png", dpi=300)
        plt.close()
        sys.exit()

        # calculate mean error for each artery and variable
        for var in vars:
            for artery in arteries:
                results[var][artery] = torch.mean(torch.stack(results[var][artery]))
        # save results in csv file
        for var in vars:
            with open(f"{config.plots.path}/{var}_errors.csv", "w") as f:
                f.write("artery,error\n")
                for artery in arteries:
                    f.write(f"{artery},{results[var][artery]}\n")
        # save full error in csv file
        with open(f"{config.plots.path}/full_errors.csv", "w") as f:
            f.write("error\n")
            f.write(f"{torch.mean(torch.stack(full))}\n")
        print(f"full = {torch.mean(torch.stack(full))}")

if __name__ == "__main__":
    main()
