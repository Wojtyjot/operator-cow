import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import wandb
from data_utils import COWDataset, MIODataLoader, WeightedLpRelLoss
from inverse.COW import COW, Multiple_COWs
from inverse.Find_RCR import find_windkessel
from log_plots import plot_predictions
from models.ae import MLAE
from models.cgpt import CGPT
from models.mmgpt import GNOT, GNOT_DISCRIMINATOR
from models.optimizer import AdamW
from models.VANO import VANO
from omegaconf import DictConfig, OmegaConf
from torch.optim.lr_scheduler import OneCycleLR
from train_new import train
from utils import seeding, utils
from utils.utils import UnitTransformer_2, get_arteries_dict
import pickle
from inverse.Find_RCR import find_windkessel

# from operatorcow.inverse.inverse import optimize_input_test

logger = logging.getLogger(__name__)

OmegaConf.register_new_resolver(
    "generate_random_seed", seeding.generate_random_seed, use_cache=True
)


@hydra.main(
    version_base=None, config_path="configs", config_name="Inverse_full_new_hyper"
)
def main(config: DictConfig) -> None:

    if config.log:
        wandb.init(
            config=OmegaConf.to_container(config, resolve=True, throw_on_missing=True),
            project=config.wandb.project,
            tags=config.wandb.tags,
            anonymous=config.wandb.anonymous,
            mode=config.wandb.mode,
            dir=Path(config.wandb.dir).absolute(),
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(2137)

    # create normalizers from data not dataset.
    normalizer_x = UnitTransformer_2(
        mean=torch.Tensor([[1.8484, 0.4340]]), std=torch.Tensor([[3.1742, 0.2565]])
    )
    # normalizer_x = UnitTransformer_2(
    #    mean=torch.Tensor([[0.5, 0.5]]), std=torch.Tensor([[0.2945, 0.2916]])
    # )
    normalizer_y = UnitTransformer_2(
        mean=torch.Tensor([[1.1976e05, 6.5500e-02, 3.1310e01]]),
        std=torch.Tensor([[1.8099e04, 5.1532e-02, 3.0624e01]]),
    )
    normalizer_theta = UnitTransformer_2(
        mean=torch.Tensor(
            [
                [
                    0.0000e00,
                    1.1111e-01,
                    5.5556e-02,
                    1.1111e-01,
                    1.1111e-01,
                    1.1111e-01,
                    1.1111e-01,
                    1.1111e-01,
                    1.1111e-01,
                    5.5556e-02,
                    1.1111e-01,
                    1.7662e01,
                    1.7706e01,
                    2.9100e00,
                    5.0039e-01,
                    5.0072e-01,
                    1.5065e00,
                    1.4986e00,
                    2.4043e00,
                    2.3981e00,
                    1.1987e00,
                    1.2037e00,
                    4.9940e-01,
                    4.9961e-01,
                    3.3058e00,
                    3.3002e00,
                    2.9983e-01,
                    4.6017e00,
                    4.5875e00,
                    1.9644e-01,
                    1.9959e-01,
                    1.6195e-01,
                    1.9981e-01,
                    2.0124e-01,
                    7.4259e-02,
                    7.2752e-02,
                    1.4357e-01,
                    1.4450e-01,
                    1.1695e-01,
                    1.1656e-01,
                    1.0671e-01,
                    1.0731e-01,
                    1.2332e-01,
                    1.2274e-01,
                    7.3721e-02,
                    1.0860e-01,
                    1.0854e-01,
                    6.9456e01,
                    1.1287e08,
                    1.2119e-08,
                ]
            ]
        ),
        std=torch.Tensor(
            [
                [
                    1.0000e-08,
                    3.1427e-01,
                    2.2906e-01,
                    3.1427e-01,
                    3.1427e-01,
                    3.1427e-01,
                    3.1427e-01,
                    3.1427e-01,
                    3.1427e-01,
                    2.2906e-01,
                    3.1427e-01,
                    1.3492e00,
                    1.3395e00,
                    2.2235e-01,
                    3.6863e-02,
                    3.6870e-02,
                    1.1427e-01,
                    1.1470e-01,
                    1.7390e-01,
                    1.7464e-01,
                    9.0034e-02,
                    8.7888e-02,
                    3.8154e-02,
                    3.7272e-02,
                    2.4628e-01,
                    2.4983e-01,
                    2.2109e-02,
                    3.4318e-01,
                    3.4641e-01,
                    3.4018e-02,
                    3.5886e-02,
                    3.1958e-02,
                    3.9021e-02,
                    3.9338e-02,
                    1.3626e-02,
                    1.4379e-02,
                    2.6881e-02,
                    2.7712e-02,
                    2.2955e-02,
                    2.3193e-02,
                    2.1227e-02,
                    2.1547e-02,
                    2.1564e-02,
                    2.2400e-02,
                    1.4715e-02,
                    1.8198e-02,
                    1.8286e-02,
                    5.7472e00,
                    1.6219e07,
                    1.1749e-08,
                ]
            ]
        ),
    )
    normalizer_u_bc = UnitTransformer_2(
        mean=torch.Tensor([[31.2255]]), std=torch.Tensor([[30.6789]])
    )

    normalizer_x = normalizer_x.to(device)
    normalizer_y = normalizer_y.to(device)
    normalizer_theta = normalizer_theta.to(device)
    normalizer_u_bc = normalizer_u_bc.to(device)

    # load models
    model_surrogate = GNOT(
        trunk_size=config.model.trunk_size,
        branch_sizes=config.model.branch_sizes,
        output_size=config.model.output_size,
        space_dim=config.model.space_dim,
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
    # AE_model = MLAE(
    #    layers=config.model.ae.layers,
    # )
    VANO_model = VANO(
        layers_encoder=config.model.VANO.layers_encoder,
        layers_decoder=config.model.VANO.layers_decoder,
        latent_dim=config.model.VANO.latent_dim,
    )
    # Load model weights
    model_surrogate = model_surrogate.to(device)
    # AE_model = AE_model.to(device)
    VANO_model = VANO_model.to(device)

    model_surrogate.load_state_dict(torch.load(config.model.surrogate_weights_path))
    # AE_model.load_state_dict(torch.load(config.model.AE_weights_path))
    VANO_model.load_state_dict(torch.load(config.model.VANO_weights_path))

    # need to iterate over folder vith validation data

    ## create arteries dict
    arteries_log = get_arteries_dict()

    val_path = Path(config.data.data_path)
    L2s = []
    i = 0
    for subfolder in val_path.iterdir():
        if subfolder.is_dir():
            arteries_log_save = get_arteries_dict()
            fig_path = (
                "/home/wssk-ptw/Operator/COW_DATASET_WRO_1_PI/WYNIKI/INVERSE_NOISE/"
                + str(subfolder.name)
            )
            rec_path = (
                "/home/wssk-ptw/Operator/COW_DATASET_WRO_1_PI/WYNIKI/INVERSE_NOISE_rec_sol/"
                + str(subfolder.name)
            )
            rec_path = Path(rec_path)

            if not Path(rec_path).exists():
                Path(rec_path).mkdir(parents=True, exist_ok=True)

            if not Path(fig_path).exists():
                Path(fig_path).mkdir(parents=True, exist_ok=True)

            d_p = str(subfolder.resolve()) + "/"

            COWs = list(
                COW(
                    model_surrogate=model_surrogate,
                    AE_model=None,
                    normalizer_x=normalizer_x,
                    normalizer_y=normalizer_y,
                    normalizer_theta=normalizer_theta,
                    device=device,
                    joints_path=config.data.joints_path,
                    lr=config.inverse.lr,
                    track=config.log,
                    data_path=d_p,
                    normalizer_u_bc=normalizer_u_bc,
                    model_VANO=VANO_model,
                    VANO=True,
                    idx=i,
                )
                for i in range(10)
            )
            M_COWs = Multiple_COWs(
                COWs,
                normalizer_x,
                normalizer_y,
                normalizer_theta,
                model_surrogate,
                config.inverse.lr,
            )

            st = time.time()
            M_COWs.solve_inverse(
                max_iters=config.inverse.max_iters,
                eps=config.inverse.eps,
                batch_size=2,
                lambda_mes=config.inverse.lambda_mes,
                lambda_mass=config.inverse.lambda_mass,
                lambda_pressure=config.inverse.lambda_pressure,
                lambda_a0=config.inverse.lambda_a0,
            )
            print(f"Time: {time.time() - st}")

            # get best cow
            R2, C, Z = find_windkessel(M_COWs.get_best(), 5)
            print("for subfolder: ", subfolder)
            print("R2: ", R2)
            print("C: ", C)
            print("Z: ", Z)
            sys.exit()
            arteries_log = M_COWs.get_validation(arteries_log)
            arteries_log_save = M_COWs.get_validation(arteries_log_save)
            L2 = M_COWs.get_L2()
            L2s.append(L2)
            M_COWs.dump_plots(fig_path)
            M_COWs.dump_params(fig_path)
            M_COWs.dump_statistics(fig_path)
            M_COWs.dump_validation(fig_path, arteries_log_save)
            M_COWs.dump_reconstructed_u_bc_plots(fig_path)
            M_COWs.dump_solutions(rec_path)
            M_COWs.dump_mesurement_plots(fig_path)

            print(f"Validation data: {subfolder}")
            print(L2)
            # sys.exit()
            i += 1
            wandb.log({"rL2": L2})

            if i == 30:
                pass

    tbl_l2 = wandb.Table(columns=["rL2_mean", "rL2_std"])
    L2s = np.array(L2s)
    tbl_l2.add_data(L2s.mean(), L2s.std())
    wandb.log({"L2": tbl_l2})

    tbl_arteries = wandb.Table(
        columns=[
            "Artery",
            "rL2 Area mean",
            "rL2 Area std",
            "rL2 Velocity mean",
            "rL2 Velocity std",
            "rL2 Flow mean",
            "rL2 Flow std",
            "rL2 Pressure mean",
            "rL2 Pressure std",
        ]
    )
    # transform lists inside nested dict to numpy arrays
    for artery in arteries_log:
        for key in arteries_log[artery]:
            arteries_log[artery][key] = np.array(arteries_log[artery][key])

    # save arteries log dict
    with open(
        "/home/wssk-ptw/Operator/COW_DATASET_WRO_1_PI/WYNIKI/INVERSE/arteries_log.pkl",
        "wb",
    ) as f:
        pickle.dump(arteries_log, f)

    # load arteries log dict
    # with open("/home/wssk-ptw/Operator/COW_DATASET_WRO_1_PI/WYNIKI/INVERSE/arteries_log.pkl", "rb") as f:
    #    arteries_log = pickle.load(f)

    for artery in arteries_log:
        tbl_arteries.add_data(
            artery,
            arteries_log[artery]["Area"].mean(),
            arteries_log[artery]["Area"].std(),
            arteries_log[artery]["Velocity"].mean(),
            arteries_log[artery]["Velocity"].std(),
            arteries_log[artery]["Flow"].mean(),
            arteries_log[artery]["Flow"].std(),
            arteries_log[artery]["Pressure"].mean(),
            arteries_log[artery]["Pressure"].std(),
        )
    wandb.log({"Arteries": tbl_arteries})

    return L2s.mean()


if __name__ == "__main__":
    L2s = main()
    wandb.log({"L2s": L2s})
