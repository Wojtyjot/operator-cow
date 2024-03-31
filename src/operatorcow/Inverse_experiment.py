import logging
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import wandb
from data_utils import COWDataset, MIODataLoader, WeightedLpRelLoss
from inverse.inverse import optimize_input_test_VANO
from log_plots import plot_predictions
from models.ae import MLAE
from models.cgpt import CGPT
from models.mmgpt import GNOT
from models.optimizer import AdamW
from models.VANO import VANO
from omegaconf import DictConfig, OmegaConf
from torch.optim.lr_scheduler import OneCycleLR
from train_new import train
from utils import seeding, utils
from utils.utils import UnitTransformer_2

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
    normalizer_x = dataset.x_normalizer.to(device)
    normalizer_u_bc = UnitTransformer_2(
        mean=torch.Tensor([[32.0895]]), std=torch.Tensor([[26.3722]])
    )
    normalizer_u_bc = normalizer_u_bc.to(device)

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
    # get data
    g, theta, inputs_f = test_data[1709]
    g, theta, inputs_f = g.to(device), theta.to(device), inputs_f.to(device)

    # optimize input
    # evaluate whole domain
    metric = WeightedLpRelLoss(
        p=2,
        component="all",
        normalizer=normalizer,
    )

    u_bc, SV_rec, out, true_in_bc, SV_true = optimize_input_test_VANO(
        model_surrogate=model_surrogate,
        VANO_model=VANO_model,
        g=g,
        inputs_f=inputs_f,
        theta=theta,
        lambda_reg=config.inverse.lambda_reg,
        max_steps=config.inverse.max_steps,
        device=device,
        normalizer_up=normalizer_up,
        normalizer_y=normalizer,
        normalizer_x=normalizer_x,
        metric=metric,
        normalizer_u_bc=normalizer_u_bc,
    )

    # evaluate whole domain
    metric = WeightedLpRelLoss(
        p=2,
        component="all",
        normalizer=normalizer,
    )

    loss_whole_domain, _, _ = metric(g, out, g.ndata["y"].squeeze())

    print(f"L2 loss whole = {loss_whole_domain.item()}")

    # Need to log plots in table
    plot_predictions(
        out.detach().cpu().numpy(), g.ndata["y"].squeeze().detach().cpu().numpy(), "3"
    )

    tbl = wandb.Table(columns=["SV_true", "SV_pred", "Loss"])
    tbl.add_data(
        SV_true,
        SV_rec,
        loss_whole_domain.item(),
    )
    wandb.log({"Rec_params": tbl})


if __name__ == "__main__":
    main()
