import logging
from pathlib import Path

import hydra
import torch
import torch.nn as nn
import wandb
from data_utils import COWDataset, MIODataLoader, WeightedLpRelLoss
from models.cgpt import CGPT
from models.mmgpt import GNOT
from models.optimizer import AdamW
from omegaconf import DictConfig, OmegaConf
from torch.optim.lr_scheduler import OneCycleLR
from train_new import train
from utils import seeding, utils

logger = logging.getLogger(__name__)

OmegaConf.register_new_resolver(
    "generate_random_seed", seeding.generate_random_seed, use_cache=True
)


@hydra.main(version_base=None, config_path="configs", config_name="test_exp")
def main(config: DictConfig) -> None:

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
    dataset = COWDataset(config.data.path)
    torch.manual_seed(2137)
    train_data, test_data = torch.utils.data.random_split(dataset, [0.8, 0.2])
    torch.manual_seed(config.seed)
    # save normalizer to renormalize data for plotting and evaluation
    normalizer = dataset.y_normalizer.to(device)
    normalizer_up = dataset.up_normalizer.to(device)

    train_loader = MIODataLoader(
        train_data,
        batch_size=config.data.batch_size,
        shuffle=True,
        drop_last=False,
    )
    test_loader = MIODataLoader(
        test_data,
        batch_size=config.data.batch_size,
        shuffle=False,
        drop_last=False,
    )

    # Load model
    if config.model.name == "CGPT":

        model = CGPT(
            trunk_size=dataset.config["input_dim"] + dataset.config["theta_dim"],
            branch_sizes=dataset.config["branch_sizes"],
            output_size=dataset.config["output_dim"],
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
            trunk_size=dataset.config["input_dim"] + dataset.config["theta_dim"],
            branch_sizes=dataset.config["branch_sizes"],
            output_size=dataset.config["output_dim"],
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

    # Put model on device
    model = model.to(device)

    # Load optimizer
    optimizer = AdamW(
        params=model.parameters(),
        lr=config.optimizer.lr,
        betas=(config.optimizer.beta1, config.optimizer.beta2),
        weight_decay=config.optimizer.weight_decay,
    )

    # Load lr_scheduler
    scheduler = OneCycleLR(
        optimizer,
        max_lr=config.optimizer.lr,
        div_factor=1e4,
        epochs=config.training.epochs,
        steps_per_epoch=len(train_loader),
    )

    # Load loss function
    loss_func = WeightedLpRelLoss(
        p=2,
        component="all",
        normalizer=normalizer,
    )

    metric_func = WeightedLpRelLoss(
        p=2,
        component="all",
        normalizer=normalizer,
    )

    result = train(
        model=model,
        loss_func=loss_func,
        metric_func=metric_func,
        train_loader=train_loader,
        val_loader=test_loader,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        epochs=config.training.epochs,
        device=device,
        grad_clip=config.training.grad_clip,
        normalizer_up=normalizer_up,
    )


if __name__ == "__main__":
    main()
