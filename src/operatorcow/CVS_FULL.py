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

# from operatorcow.inverse.inverse import optimize_input_test

logger = logging.getLogger(__name__)

OmegaConf.register_new_resolver(
    "generate_random_seed", seeding.generate_random_seed, use_cache=True
)


@hydra.main(version_base=None, config_path="configs", config_name="Inverse_full_new_hyper")
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
        mean=torch.Tensor([[1.1976e+05, 6.5500e-02, 3.1310e+01]]),
        std=torch.Tensor([[1.8099e+04, 5.1532e-02, 3.0624e+01]]),
    )
    normalizer_theta = UnitTransformer_2(
        mean=torch.Tensor(
           [[0.0000e+00, 1.1111e-01, 5.5556e-02, 1.1111e-01, 1.1111e-01, 1.1111e-01,1.1111e-01, 1.1111e-01, 1.1111e-01, 5.5556e-02, 1.1111e-01, 1.7662e+01,
         1.7706e+01, 2.9100e+00, 5.0039e-01, 5.0072e-01, 1.5065e+00, 1.4986e+00,
         2.4043e+00, 2.3981e+00, 1.1987e+00, 1.2037e+00, 4.9940e-01, 4.9961e-01,
         3.3058e+00, 3.3002e+00, 2.9983e-01, 4.6017e+00, 4.5875e+00, 1.9644e-01,
         1.9959e-01, 1.6195e-01, 1.9981e-01, 2.0124e-01, 7.4259e-02, 7.2752e-02,
         1.4357e-01, 1.4450e-01, 1.1695e-01, 1.1656e-01, 1.0671e-01, 1.0731e-01,
         1.2332e-01, 1.2274e-01, 7.3721e-02, 1.0860e-01, 1.0854e-01, 6.9456e+01,
         1.1287e+08, 1.2119e-08]]
        ),
        std=torch.Tensor(
           [[1.0000e-08, 3.1427e-01, 2.2906e-01, 3.1427e-01, 3.1427e-01, 3.1427e-01, 3.1427e-01, 3.1427e-01, 3.1427e-01, 2.2906e-01, 3.1427e-01, 1.3492e+00,
         1.3395e+00, 2.2235e-01, 3.6863e-02, 3.6870e-02, 1.1427e-01, 1.1470e-01,
         1.7390e-01, 1.7464e-01, 9.0034e-02, 8.7888e-02, 3.8154e-02, 3.7272e-02,
         2.4628e-01, 2.4983e-01, 2.2109e-02, 3.4318e-01, 3.4641e-01, 3.4018e-02,
         3.5886e-02, 3.1958e-02, 3.9021e-02, 3.9338e-02, 1.3626e-02, 1.4379e-02,
         2.6881e-02, 2.7712e-02, 2.2955e-02, 2.3193e-02, 2.1227e-02, 2.1547e-02,
         2.1564e-02, 2.2400e-02, 1.4715e-02, 1.8198e-02, 1.8286e-02, 5.7472e+00,
         1.6219e+07, 1.1749e-08]]
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

    cvs_path = Path(config.data.cvs_path) #path to cvs
    rec_folder = Path("/home/wssk-ptw/Operator/COW_DATASET_WRO_1_PI/Rec_CVS/")
    fig_path = Path("/home/wssk-ptw/Operator/COW_DATASET_WRO_1_PI/WYNIKI/CVS/")
    if not os.path.exists(rec_folder):
        os.makedirs(rec_folder)
    # setup dictionaries for arteries error measurements
    
    arteries_1_1 = get_arteries_dict()
    arteries_1_2 = get_arteries_dict()
    arteries_1_3 = get_arteries_dict()
    arteries_2_1 = get_arteries_dict()
    arteries_2_2 = get_arteries_dict()
    arteries_2_3 = get_arteries_dict()
    arteries_3_1 = get_arteries_dict()
    arteries_3_2 = get_arteries_dict()
    arteries_3_3 = get_arteries_dict()
    arteries_4_1 = get_arteries_dict()
    arteries_4_2 = get_arteries_dict()
    arteries_4_3 = get_arteries_dict()
    arteries_5_1 = get_arteries_dict()
    arteries_5_2 = get_arteries_dict()
    arteries_5_3 = get_arteries_dict()
    arteries_6_1 = get_arteries_dict()
    arteries_6_2 = get_arteries_dict()
    arteries_6_3 = get_arteries_dict()
    arteries_7_1 = get_arteries_dict()
    arteries_7_2 = get_arteries_dict()
    arteries_7_3 = get_arteries_dict()
    arteries_dict = {
        "1_1": arteries_1_1,
        "1_2": arteries_1_2,
        "1_3": arteries_1_3,
        "2_1": arteries_2_1,
        "2_2": arteries_2_2,
        "2_3": arteries_2_3,
        "3_1": arteries_3_1,
        "3_2": arteries_3_2,
        "3_3": arteries_3_3,
        "4_1": arteries_4_1,
        "4_2": arteries_4_2,
        "4_3": arteries_4_3,
        "5_1": arteries_5_1,
        "5_2": arteries_5_2,
        "5_3": arteries_5_3,
        "6_1": arteries_6_1,
        "6_2": arteries_6_2,
        "6_3": arteries_6_3,
        "7_1": arteries_7_1,
        "7_2": arteries_7_2,
        "7_3": arteries_7_3,
    }
   
    L2_ref = []
    L2_1_1 = []
    L2_1_2 = []
    L2_1_3 = []
    L2_2_1 = []
    L2_2_2 = []
    L2_2_3 = []
    L2_3_1 = []
    L2_3_2 = []
    L2_3_3 = []
    L2_4_1 = []
    L2_4_2 = []
    L2_4_3 = []
    L2_5_1 = []
    L2_5_2 = []
    L2_5_3 = []
    L2_6_1 = []
    L2_6_2 = []
    L2_6_3 = []
    L2_7_1 = []
    L2_7_2 = []
    L2_7_3 = []

    L2_dict = {
        "ref": L2_ref,
        "1_1": L2_1_1,
        "1_2": L2_1_2,
        "1_3": L2_1_3,
        "2_1": L2_2_1,
        "2_2": L2_2_2,
        "2_3": L2_2_3,
        "3_1": L2_3_1,
        "3_2": L2_3_2,
        "3_3": L2_3_3,
        "4_1": L2_4_1,
        "4_2": L2_4_2,
        "4_3": L2_4_3,
        "5_1": L2_5_1,
        "5_2": L2_5_2,
        "5_3": L2_5_3,
        "6_1": L2_6_1,
        "6_2": L2_6_2,
        "6_3": L2_6_3,
        "7_1": L2_7_1,
        "7_2": L2_7_2,
        "7_3": L2_7_3,
    }


    for con_type in cvs_path.iterdir():
        cow_path = cvs_path / con_type
        for cow in cow_path.iterdir():
            con_path = cow_path / cow
            for sub in ["ref", "1", "2", "3"]:
                if sub == "ref":
                    ### here create cows and solve, and dump ref
                    log_arteries = get_arteries_dict()
                    path_to_ref = con_path / sub
                    fig = fig_path / con_type / cow / sub
                    if not os.path.exists(fig):
                        os.makedirs(fig)

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
                        data_path=path_to_ref,
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
                    M_COWs.solve_inverse(
                    max_iters=config.inverse.max_iters,
                    eps=config.inverse.eps,
                    batch_size=2,
                    lambda_mes=config.inverse.lambda_mes,
                    lambda_mass=config.inverse.lambda_mass,
                    lambda_pressure=config.inverse.lambda_pressure,
                    lambda_a0=config.inverse.lambda_a0,
                    )

                    initial_run_ref = con_path / "initial_run_ref"
                    if not os.path.exists(initial_run_ref):
                        os.makedirs(initial_run_ref)

                    
                    M_COWs.dump_ref_for_cvs(initial_run_ref)
                    L2 = M_COWs.get_L2()
                    L2_dict["ref"].append(L2)
                    M_COWs.dump_plots(fig)
                    M_COWs.dump_params(fig)
                    M_COWs.dump_statistics(fig)
                    M_COWs.get_validation(log_arteries)
                    M_COWs.dump_validation(fig, log_arteries)
                    M_COWs.dump_reconstructed_u_bc_plots(fig)

                    if config.log:
                        wandb.log({"rL2 ref": L2})


                    #Need to sve ref
                
                else:
                    # here solve cvs case and dump solutions
                    path_to_cvs = con_path / sub
                    fig = fig_path / con_type / cow / sub
                    if not os.path.exists(fig):
                        os.makedirs(fig)
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
                            data_path=path_to_cvs,
                            normalizer_u_bc=normalizer_u_bc,
                            model_VANO=VANO_model,
                            VANO=True,
                            cvs=True,
                            r0s_path=path_to_ref / "r0s.npy",
                            p_ref_path=path_to_ref,
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
                    
                    # DUMP SOLUTIONS !!!! TODO
                    rec_path = rec_folder / con_type / cow / sub

                    log_save = get_arteries_dict()
                    M_COWs.dump_solutions(rec_path)
                    arteries_dict[f"{con_type}_{sub}"] = M_COWs.get_validation(arteries_dict[f"{con_type}_{sub}"])
                    L2 = M_COWs.get_L2()
                    L2_dict[f"{con_type}_{sub}"].append(L2)
                    M_COWs.dump_plots(fig)
                    M_COWs.dump_params(fig)
                    M_COWs.dump_statistics(fig)
                    M_COWs.get_validation(log_save)
                    M_COWs.dump_validation(fig, log_save)
                    M_COWs.dump_reconstructed_u_bc_plots(fig)

    if config.log:
        for key, value in L2_dict.items():
            wandb.log({f"rL2 {key}": value})
        for key, value in arteries_dict.items():
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
            # transform into numpy arrays
            for artery in arteries_dict[key]:
                for modality in arteries_dict[key][artery]:
                    arteries_dict[key][artery][modality] = np.array(arteries_dict[key][artery][modality])

            #save to disc
            with open(fig_path / f"{key}_arteries_dict.pkl", "wb") as f:
                pickle.dump(arteries_dict[key], f)
            
            # save to wandb
            for artery in arteries_dict[key]:
                tbl_arteries.add_data(
                    artery,
                    arteries_dict[key][artery]["Area"].mean(),
                    arteries_dict[key][artery]["Area"].std(),
                    arteries_dict[key][artery]["Velocity"].mean(),
                    arteries_dict[key][artery]["Velocity"].std(),
                    arteries_dict[key][artery]["Flow"].mean(),
                    arteries_dict[key][artery]["Flow"].std(),
                    arteries_dict[key][artery]["Pressure"].mean(),
                    arteries_dict[key][artery]["Pressure"].std(),
                )
            wandb.log({f"{key}_arteries": tbl_arteries})
                    
