import logging
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import wandb
from data_utils import COWDataset, MIODataLoader, WeightedLpRelLoss
from inverse.COW import COW
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
from utils.utils import UnitTransformer_2

# from operatorcow.inverse.inverse import optimize_input_test

logger = logging.getLogger(__name__)

OmegaConf.register_new_resolver(
    "generate_random_seed", seeding.generate_random_seed, use_cache=True
)


@hydra.main(version_base=None, config_path="configs", config_name="inverse_full")
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

    torch.manual_seed(config.seed)

    # create normalizers from data not dataset.
    normalizer_x = UnitTransformer_2(
        mean=torch.Tensor([[1.8504, 0.4671]]), std=torch.Tensor([[3.1793, 0.2733]])
    )
    normalizer_y = UnitTransformer_2(
        mean=torch.Tensor([[1.2793e+05, 6.2301e-02, 2.5150e+01]]),
        std=torch.Tensor([[2.3825e+04, 4.3402e-02, 2.3625e+01]]),
    )
    normalizer_theta = UnitTransformer_2(
        mean=torch.Tensor(
            [[0.0000e+00, 1.1111e-01, 5.5556e-02, 1.1111e-01, 1.1111e-01, 1.1111e-01,
         1.1111e-01, 1.1111e-01, 1.1111e-01, 5.5556e-02, 1.1111e-01, 1.7711e+01,
         1.7697e+01, 2.9023e+00, 4.9929e-01, 4.9919e-01, 1.4972e+00, 1.5071e+00,
         2.4015e+00, 2.3984e+00, 1.2007e+00, 1.1987e+00, 5.0048e-01, 5.0026e-01,
         3.2972e+00, 3.3060e+00, 3.0045e-01, 4.5951e+00, 4.6005e+00, 2.0034e-01,
         2.0103e-01, 1.6219e-01, 2.0010e-01, 1.9990e-01, 7.2940e-02, 7.2973e-02,
         1.4333e-01, 1.4283e-01, 1.1701e-01, 1.1733e-01, 1.0679e-01, 1.0718e-01,
         1.1981e-01, 1.1998e-01, 7.4069e-02, 1.0483e-01, 1.0494e-01, 6.4335e+01,
         1.3745e+08, 9.9070e-09]]
        ),
        std=torch.Tensor(
            [[1.0000e-08, 3.1427e-01, 2.2906e-01, 3.1427e-01, 3.1427e-01, 3.1427e-01,
         3.1427e-01, 3.1427e-01, 3.1427e-01, 2.2906e-01, 3.1427e-01, 1.2840e+00,
         1.3337e+00, 2.1418e-01, 3.6259e-02, 3.7161e-02, 1.1344e-01, 1.1346e-01,
         1.7721e-01, 1.8211e-01, 8.9553e-02, 8.9506e-02, 3.7230e-02, 3.7044e-02,
         2.5030e-01, 2.4927e-01, 2.2290e-02, 3.4071e-01, 3.4680e-01, 1.9805e-02,
         1.9342e-02, 1.6200e-02, 1.9381e-02, 2.0292e-02, 7.2833e-03, 7.4511e-03,
         1.4521e-02, 1.4617e-02, 1.1485e-02, 1.1885e-02, 1.0993e-02, 1.0710e-02,
         1.2050e-02, 1.2239e-02, 7.3845e-03, 1.0549e-02, 1.0472e-02, 2.6040e+00,
         1.7545e+07, 1.1245e-08]]
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

    # initialize COW
    cow = COW(
        model_surrogate=model_surrogate,
        AE_model=None,
        normalizer_x=normalizer_x,
        normalizer_y=normalizer_y,
        normalizer_theta=normalizer_theta,
        device=device,
        joints_path=config.data.joints_path,
        lr=config.inverse.lr,
        track=config.log,
        data_path=config.data.data_path,
        normalizer_u_bc=normalizer_u_bc,
        model_VANO=VANO_model,
        VANO=True,
    )

    L2 = cow.solve_accumulate(
        max_iters=config.inverse.max_iters,
        eps=config.inverse.eps,
        batch_size=config.inverse.batch_size,
        lambda_reg=config.inverse.lambda_reg,
        lambda_bif=config.inverse.lambda_bif,
        lambda_sv=config.inverse.lambda_sv,
        lambda_mes=config.inverse.lambda_mes,
        log_every=config.inverse.log_every,
    )

    print(L2)


if __name__ == "__main__":
    main()
