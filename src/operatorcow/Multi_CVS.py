import logging
import sys
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

# from operatorcow.inverse.inverse import optimize_input_test

logger = logging.getLogger(__name__)

OmegaConf.register_new_resolver(
    "generate_random_seed", seeding.generate_random_seed, use_cache=True
)


@hydra.main(version_base=None, config_path="configs", config_name="cvs_mca")
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
        mean=torch.Tensor([[1.8504, 0.4671]]), std=torch.Tensor([[3.1793, 0.2733]])
    )
    # normalizer_x = UnitTransformer_2(
    #    mean=torch.Tensor([[0.5, 0.5]]), std=torch.Tensor([[0.2945, 0.2916]])
    # )
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

    path_15386 = config.data.path_15386
    r0s_path = config.data.r0s_path
    p_ref_path = config.data.p_ref_path
    cvs_path = config.data.cvs_path

    L2s = []
    # i = 0

    arteries_log_save = get_arteries_dict()
    # get 10 runs for cvs and compute losses

    fig_path = f"/home/wssk-ptw/Operator/COW_DATASET/WYNIKI/CVS_MCA_{0}/" + "NO_CVS"
    p_ref_path = f"/home/wssk-ptw/Operator/COW_DATASET/CVS_MCA_{0}/p_ref/"
    r0s_path = f"/home/wssk-ptw/Operator/COW_DATASET/CVS_MCA_{0}/r0s/"
    if not Path(fig_path).exists():
        Path(fig_path).mkdir(parents=True, exist_ok=True)
    if not Path(p_ref_path).exists():
        Path(p_ref_path).mkdir(parents=True, exist_ok=True)
    if not Path(r0s_path).exists():
        Path(r0s_path).mkdir(parents=True, exist_ok=True)
    if True:

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
                data_path=path_15386,
                normalizer_u_bc=normalizer_u_bc,
                model_VANO=VANO_model,
                VANO=True,
            )
            for _ in range(10)
        )
        M_COWs = Multiple_COWs(
            COWs,
            normalizer_x,
            normalizer_y,
            normalizer_theta,
            model_surrogate,
            config.inverse.lr,
        )
        M_COWs.solve_inverse(
            max_iters=config.inverse.max_iters,
            eps=config.inverse.eps,
            batch_size=2,
            lambda_mes=config.inverse.lambda_mes,
            lambda_mass=config.inverse.lambda_mass,
            lambda_pressure=config.inverse.lambda_pressure,
            lambda_a0=config.inverse.lambda_a0,
            dump_p_ref=True,
        )

        arteries_log = M_COWs.get_validation(arteries_log)
        arteries_log_save = M_COWs.get_validation(arteries_log_save)
        L2 = M_COWs.get_L2()
        L2s.append(L2)
        M_COWs.dump_plots(fig_path)
        M_COWs.dump_params(fig_path)
        M_COWs.dump_statistics(fig_path)
        M_COWs.dump_validation(fig_path, arteries_log_save)
        M_COWs.dump_reconstructed_u_bc_plots(fig_path)

        #print(f"Validation data: {subfolder}")
        #print(L2)
        # sys.exit()

        # if i == 10:
        #    break
        if config.log:
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

    for subfolder in ["CVS_0.9", "CVS_0.7", "CVS_0.5"]:
        arteries_log = get_arteries_dict()
        arteries_log_save = get_arteries_dict()
        fig_path = f"/home/wssk-ptw/Operator/COW_DATASET/WYNIKI/CVS_MCA_{0}/" + str(
            subfolder
        )
        if not Path(fig_path).exists():
            Path(fig_path).mkdir(parents=True, exist_ok=True)
        L2s = []
        d_p = cvs_path + subfolder + "/"
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
                cvs=True,
                r0s_path=r0s_path + "r0s.npy",
                p_ref_path=p_ref_path,
            )
            for _ in range(10)
        )
        M_COWs = Multiple_COWs(
            COWs,
            normalizer_x,
            normalizer_y,
            normalizer_theta,
            model_surrogate,
            config.inverse.lr,
        )
        M_COWs.solve_cvs(
            max_iters=config.inverse.max_iters,
            eps=config.inverse.eps,
            batch_size=2,
            lambda_mes=config.inverse.lambda_mes,
            lambda_mass=config.inverse.lambda_mass,
            lambda_pressure=config.inverse.lambda_pressure,
            lambda_a0=config.inverse.lambda_a0,
            lambda_p_ref=config.inverse.lambda_p_ref,
        )

        arteries_log = M_COWs.get_validation(arteries_log)
        arteries_log_save = M_COWs.get_validation(arteries_log_save)
        L2 = M_COWs.get_L2()
        L2s.append(L2)
        M_COWs.dump_plots(fig_path)
        M_COWs.dump_params(fig_path)
        M_COWs.dump_statistics(fig_path)
        M_COWs.dump_validation(fig_path, arteries_log_save)
        M_COWs.dump_reconstructed_u_bc_plots(fig_path)

        print(f"Validation data: {subfolder}")
        print(L2)

        if config.log:
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
            wandb.log({f"Arteries{subfolder}": tbl_arteries})

    return L2s.mean()


if __name__ == "__main__":
    L2s = main()
    wandb.log({"L2s": L2s})
