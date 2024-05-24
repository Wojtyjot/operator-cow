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
from inverse.COW import COW
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
    # print(wandb.wandb.config.trunk_size)

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
        mean=torch.Tensor([[1.8504, 0.4671]]), std=torch.Tensor([[3.1793, 0.2733]])
    )
    normalizer_y = UnitTransformer_2(
        mean=torch.Tensor([[1.2793e05, 6.2301e-02, 2.5150e01]]),
        std=torch.Tensor([[2.3825e04, 4.3402e-02, 2.3625e01]]),
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
                    1.7711e01,
                    1.7697e01,
                    2.9023e00,
                    4.9929e-01,
                    4.9919e-01,
                    1.4972e00,
                    1.5071e00,
                    2.4015e00,
                    2.3984e00,
                    1.2007e00,
                    1.1987e00,
                    5.0048e-01,
                    5.0026e-01,
                    3.2972e00,
                    3.3060e00,
                    3.0045e-01,
                    4.5951e00,
                    4.6005e00,
                    2.0034e-01,
                    2.0103e-01,
                    1.6219e-01,
                    2.0010e-01,
                    1.9990e-01,
                    7.2940e-02,
                    7.2973e-02,
                    1.4333e-01,
                    1.4283e-01,
                    1.1701e-01,
                    1.1733e-01,
                    1.0679e-01,
                    1.0718e-01,
                    1.1981e-01,
                    1.1998e-01,
                    7.4069e-02,
                    1.0483e-01,
                    1.0494e-01,
                    6.4335e01,
                    1.3745e08,
                    9.9070e-09,
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
                    1.2840e00,
                    1.3337e00,
                    2.1418e-01,
                    3.6259e-02,
                    3.7161e-02,
                    1.1344e-01,
                    1.1346e-01,
                    1.7721e-01,
                    1.8211e-01,
                    8.9553e-02,
                    8.9506e-02,
                    3.7230e-02,
                    3.7044e-02,
                    2.5030e-01,
                    2.4927e-01,
                    2.2290e-02,
                    3.4071e-01,
                    3.4680e-01,
                    1.9805e-02,
                    1.9342e-02,
                    1.6200e-02,
                    1.9381e-02,
                    2.0292e-02,
                    7.2833e-03,
                    7.4511e-03,
                    1.4521e-02,
                    1.4617e-02,
                    1.1485e-02,
                    1.1885e-02,
                    1.0993e-02,
                    1.0710e-02,
                    1.2050e-02,
                    1.2239e-02,
                    7.3845e-03,
                    1.0549e-02,
                    1.0472e-02,
                    2.6040e00,
                    1.7545e07,
                    1.1245e-08,
                ]
            ]
        ),
    )
    normalizer_u_bc = UnitTransformer_2(
        mean=torch.Tensor([[25.1235]]), std=torch.Tensor([[23.7002]])
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
            fig_path = "/home/wssk-ptw/Operator/COW_DATASET/WYNIKI/" + str(
                subfolder.name
            )
            if not Path(fig_path).exists():
                Path(fig_path).mkdir(parents=True, exist_ok=True)

            d_p = str(subfolder.resolve()) + "/"
            cow = COW(
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

            L2 = cow.solve_accumulate_2(
                max_iters=wandb.config.max_iters,
                eps=wandb.config.eps,
                batch_size=wandb.config.batch_size,
                lambda_mes=wandb.config.lambda_mes,
                lambda_mass=wandb.config.lambda_mass,
                lambda_pressure=wandb.config.lambda_pressure,
                lambda_a0=wandb.config.lambda_a0,
            )
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

            arteries_log = cow.get_validation(arteries_log)
            arteries_log_save = cow.get_validation(arteries_log_save)
            L2s.append(L2)
            cow.dump_plots(fig_path)
            cow.dump_params(fig_path)
            cow.dump_statistics(fig_path)
            cow.dump_validation(fig_path, arteries_log_save)
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
