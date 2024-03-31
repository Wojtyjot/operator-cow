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
        mean=torch.Tensor([[2.0742, 0.5000]]), std=torch.Tensor([[3.3294, 0.2916]])
    )
    normalizer_y = UnitTransformer_2(
        mean=torch.Tensor([[1.1865e05, 6.1614e-02, 3.2138e01]]),
        std=torch.Tensor([[2.9567e04, 4.1075e-02, 2.6303e01]]),
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
                    1.7730e01,
                    1.7725e01,
                    2.9040e00,
                    4.9763e-01,
                    4.9880e-01,
                    1.4992e00,
                    1.5056e00,
                    2.4022e00,
                    2.4074e00,
                    1.2005e00,
                    1.1999e00,
                    4.9963e-01,
                    4.9966e-01,
                    3.3025e00,
                    3.2959e00,
                    2.9919e-01,
                    8.6087e00,
                    8.5945e00,
                    2.0086e-01,
                    2.0012e-01,
                    1.6210e-01,
                    2.0003e-01,
                    1.9992e-01,
                    7.3011e-02,
                    7.3063e-02,
                    1.4271e-01,
                    1.4297e-01,
                    1.1684e-01,
                    1.1700e-01,
                    1.0681e-01,
                    1.0699e-01,
                    1.1992e-01,
                    1.1973e-01,
                    7.3971e-02,
                    1.0492e-01,
                    1.0506e-01,
                    1.2307e02,
                ]
            ]
        ),
        std=torch.Tensor(
            [
                [
                    1.0000e-08,
                    3.1428e-01,
                    2.2907e-01,
                    3.1428e-01,
                    3.1428e-01,
                    3.1428e-01,
                    3.1428e-01,
                    3.1428e-01,
                    3.1428e-01,
                    2.2907e-01,
                    3.1428e-01,
                    8.8919e-01,
                    8.8508e-01,
                    1.3855e-01,
                    2.5052e-02,
                    2.5174e-02,
                    7.5095e-02,
                    7.6750e-02,
                    1.2210e-01,
                    1.2138e-01,
                    6.0034e-02,
                    5.9057e-02,
                    2.5083e-02,
                    2.4387e-02,
                    1.6181e-01,
                    1.5916e-01,
                    1.4958e-02,
                    4.2719e-01,
                    4.3104e-01,
                    9.8974e-03,
                    1.0155e-02,
                    8.2288e-03,
                    1.0284e-02,
                    1.0545e-02,
                    3.7212e-03,
                    3.5598e-03,
                    7.0221e-03,
                    6.8241e-03,
                    5.7332e-03,
                    5.8618e-03,
                    5.1393e-03,
                    5.4781e-03,
                    6.1554e-03,
                    6.0671e-03,
                    3.6297e-03,
                    5.3287e-03,
                    5.2427e-03,
                    1.1474e01,
                ]
            ]
        ),
    )
    normalizer_u_bc = UnitTransformer_2(
        mean=torch.Tensor([[32.0895]]), std=torch.Tensor([[26.3722]])
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
