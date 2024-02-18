import logging
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import wandb
from data_utils import COWDataset, MIODataLoader, WeightedLpRelLoss
from inverse import optimize_input_test
from log_plots import plot_predictions
from models.ae import MLAE
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


@hydra.main(version_base=None, config_path="configs", config_name="inverse_exp")
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

    test_loader = MIODataLoader(
        test_data,
        batch_size=config.data.batch_size,
        shuffle=False,
        drop_last=False,
    )
    # 4500 test samples
    # for test choose sample 2137

    # Load model

    model_surrogate = GNOT(
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
    AE_model = MLAE(
        layers=config.model.ae.layers,
    )

    # Load model weights
    model_surrogate = model_surrogate.to(device)
    AE_model = AE_model.to(device)

    model_surrogate.load_state_dict(torch.load(config.model.surrogate_weights_path))
    AE_model.load_state_dict(torch.load(config.model.AE_weights_path))

    # get data
    g, inputs_f, theta = test_data[2137]
    g = g.to(device)

    # optimize input
    u_bc, SV_rec, out, true_in_bc, SV_true = optimize_input_test(
        model_surrogate,
        AE_model,
        g,
        inputs_f,
        theta,
        config.inverse.lambda_reg,
        config.inverse.max_steps,
        device,
    )

    # evaluate whole domain
    metric = WeightedLpRelLoss(
        p=2,
        component="all",
        normalizer=normalizer,
    )

    loss_whole_domain, _, _ = metric(g, out, g.ndata["y"].squeeze())

    # Need to log plots in table
    plot_predictions(out, g.ndata["y"].squeeze(), "2137")

    tbl = wandb.Table(collumns=["BC_true", "BC_pred", "SV_true", "SV_pred", "Loss"])
    tbl.add_data(
        wandb.Image(plt.plot(u_bc, label="BC_true")),
        wandb.Image(plt.plot(true_in_bc[:, -1], label="BC_pred")),
        SV_true,
        SV_rec,
        loss_whole_domain.item(),
    )


if __name__ == "__main__":
    main()
