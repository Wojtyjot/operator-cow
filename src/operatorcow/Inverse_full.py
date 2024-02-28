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
from models.mmgpt import GNOT
from models.optimizer import AdamW
from omegaconf import DictConfig, OmegaConf
from torch.optim.lr_scheduler import OneCycleLR
from train_new import train
from utils import seeding, utils
from utils.utils import UnitTransformer_2

from operatorcow.inverse.inverse import optimize_input_test

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
        mean=torch.Tensor([[0.5000]]), std=torch.Tensor([[0.2916]])
    )
    normalizer_y = UnitTransformer_2(
        mean=torch.Tensor([[32.0895]]), std=torch.Tensor([[26.3722]])
    )
    normalizer_theta = UnitTransformer_2(
        mean=torch.Tensor(
            [
                [
                    0.0000,
                    0.1111,
                    0.0556,
                    0.1111,
                    0.1111,
                    0.1111,
                    0.1111,
                    0.1111,
                    0.1111,
                    0.0556,
                    0.1111,
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
                ]
            ]
        ),
    )

    normalizer_x = normalizer_x.to(device)
    normalizer_y = normalizer_y.to(device)
    normalizer_theta = normalizer_theta.to(device)

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
    AE_model = MLAE(
        layers=config.model.ae.layers,
    )

    # Load model weights
    model_surrogate = model_surrogate.to(device)
    AE_model = AE_model.to(device)

    model_surrogate.load_state_dict(torch.load(config.model.surrogate_weights_path))
    AE_model.load_state_dict(torch.load(config.model.AE_weights_path))

    # initialize COW
    cow = COW(
        model_surrogate=model_surrogate,
        AE_model=AE_model,
        normalizer_x=normalizer_x,
        normalizer_y=normalizer_y,
        normalizer_theta=normalizer_theta,
        device=device,
        joints_path=config.data.joints_path,
        lr=config.inverse.lr,
    )

    L2 = cow.solve(
        max_iters=config.inverse.max_iters,
        eps=config.inverse.eps,
        batch_size=config.inverse.batch_size,
        lambda_reg=config.inverse.lambda_reg,
        lambda_bif=config.inverse.lambda_bif,
        lambda_sv=config.inverse.lambda_sv,
        lamda_mes=config.inverse.lamda_mes,
    )

    print(L2)


if __name__ == "__main__":
    main()
