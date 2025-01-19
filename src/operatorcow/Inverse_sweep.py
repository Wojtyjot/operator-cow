import logging
import sys
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import wandb
import yaml
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

# from operatorcow.inverse import optimize_input_test

# logger = logging.getLogger(__name__)

# OmegaConf.register_new_resolver(
#    "generate_random_seed", seeding.generate_random_seed, use_cache=True
# )


# @hydra.main(version_base=None, wandb.config_path="wandb.configs", wandb.config_name="sweep_wandb.config")
def main() -> None:
    with open(
        "/home/wssk-ptw/Operator/operator-cow/src/operatorcow/configs/sweep_config.yaml"
    ) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    run = wandb.init(config=config)
    # print(wandb.config.trunk_size)

    # if wandb.config.log:
    #    wandb.init(
    #        wandb.config=OmegaConf.to_container(wandb.config, resolve=True, throw_on_missing=True),
    #        project=wandb.config.wandb.project,
    #        tags=wandb.config.wandb.tags,
    #        anonymous=wandb.config.wandb.anonymous,
    #        mode=wandb.config.wandb.mode,
    #        dir=Path(wandb.config.wandb.dir).absolute(),
    #    )

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
        trunk_size=wandb.config.trunk_size,
        branch_sizes=wandb.config.branch_sizes,
        output_size=wandb.config.output_size,
        space_dim=wandb.config.space_dim,
        n_layers=wandb.config.n_layers,
        n_hidden=wandb.config.n_hidden,
        n_head=wandb.config.n_head,
        n_experts=wandb.config.n_experts,
        n_inner=wandb.config.n_inner,
        mlp_layers=wandb.config.mlp_layers,
        attn_type=wandb.config.attn_type,
        act=wandb.config.act,
        ffn_dropout=wandb.config.ffn_dropout,
        attn_dropout=wandb.config.attn_dropout,
        horiz_fourier_dim=wandb.config.horiz_fourier_dim,
    )
    # AE_model = MLAE(
    #    layers=wandb.config.ae.layers,
    # )
    VANO_model = VANO(
        layers_encoder=wandb.config.layers_encoder,
        layers_decoder=wandb.config.layers_decoder,
        latent_dim=wandb.config.latent_dim,
    )
    # Load model weights
    model_surrogate = model_surrogate.to(device)
    # AE_model = AE_model.to(device)
    VANO_model = VANO_model.to(device)

    model_surrogate.load_state_dict(torch.load(wandb.config.surrogate_weights_path))
    # AE_model.load_state_dict(torch.load(wandb.config.AE_weights_path))
    VANO_model.load_state_dict(torch.load(wandb.config.VANO_weights_path))

    # need to iterate over folder vith validation data

    ## create arteries dict
    arteries_log = get_arteries_dict()

    val_path = Path(wandb.config.data_path)
    L2s = []
    i = 0
    for subfolder in val_path.iterdir():
        if subfolder.is_dir():
            arteries_log_save = get_arteries_dict()
            fig_path = "/home/wssk-ptw/Operator/COW_DATASET/NEW_DATASET_WYNIKI/" + str(
                subfolder.name
            )
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
                    joints_path=wandb.config.joints_path,
                    lr=wandb.config.lr,
                    track=False,
                    data_path=d_p,
                    normalizer_u_bc=normalizer_u_bc,
                    model_VANO=VANO_model,
                    VANO=True,
                )
                for _ in range(10)
            )

            # cow = COW(
            #    model_surrogate=model_surrogate,
            #    AE_model=None,
            #    normalizer_x=normalizer_x,
            #    normalizer_y=normalizer_y,
            #    normalizer_theta=normalizer_theta,
            #    device=device,
            #    joints_path=wandb.config.joints_path,
            #    lr=wandb.config.lr,
            #    track=False,
            #    data_path=d_p,
            #    normalizer_u_bc=normalizer_u_bc,
            #    model_VANO=VANO_model,
            #    VANO=True,
            # )
            M_COWs = Multiple_COWs(
                COWs,
                normalizer_x,
                normalizer_y,
                normalizer_theta,
                model_surrogate,
                wandb.config.lr,
            )
            M_COWs.solve_inverse(
                max_iters=wandb.config.max_iters,
                eps=wandb.config.eps,
                batch_size=2,
                lambda_mes=wandb.config.lambda_mes,
                lambda_mass=wandb.config.lambda_mass,
                lambda_pressure=wandb.config.lambda_pressure,
                lambda_a0=wandb.config.lambda_a0,
            )
            L2 = M_COWs.get_L2()
            # L2 = cow.solve_accumulate_2(
            #    max_iters=wandb.config.max_iters,
            #    eps=wandb.config.eps,
            #    batch_size=wandb.config.batch_size,
            #    lambda_mes=wandb.config.lambda_mes,
            #    lambda_mass=wandb.config.lambda_mass,
            #    lambda_pressure=wandb.config.lambda_pressure,
            #    lambda_a0=wandb.config.lambda_a0,
            # )
            # estimate windkessel and perform simulation etc.
            # R2, C, Z = find_windkessel(
            #    cow, 5
            # )  # need to check for opimal num_simulations
            # 5 in paper ML in cardiovascular flows
            # TODO zmodyfikować csv BY były dobre jointy

            # cow.ROM_simulation(wandb.config.path_to_csv, 1,1, 1)
            # cow.set_ROM_mesurement()
            # another loop of training with new "mesurements"

            # L2 = cow.solve_accumulate_2(
            #    max_iters=wandb.config.max_iters,
            #    eps=wandb.config.eps,
            #    batch_size=wandb.config.batch_size,
            #    lambda_reg=wandb.config.lambda_reg,
            #    lambda_bif=wandb.config.lambda_bif,
            #    lambda_sv=wandb.config.lambda_sv,
            #    lambda_mes=wandb.config.lambda_mes,
            #    log_every=wandb.config.log_every,
            # )

            # arteries_log = cow.get_validation(arteries_log)
            # arteries_log_save = cow.get_validation(arteries_log_save)
            L2s.append(L2)
            # cow.dump_plots(fig_path)
            # cow.dump_params(fig_path)
            # cow.dump_statistics(fig_path)
            # cow.dump_validation(fig_path, arteries_log_save)
            # cow.dump_ROM_plots(fig_path)

            print(f"Validation data: {subfolder}")
            print(L2)
            # sys.exit()
            i += 1

            if i == 10:
                break

    tbl_l2 = wandb.Table(columns=["rL2_mean", "rL2_std"])
    L2s = np.array(L2s)
    tbl_l2.add_data(L2s.mean(), L2s.std())
    wandb.log({"L2": tbl_l2})

    return L2s.mean()


if __name__ == "__main__":
    L2s = main()
    wandb.log({"L2s": L2s})
