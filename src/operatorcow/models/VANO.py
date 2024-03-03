from typing import *

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class VANO(nn.Module):
    """
    Class representing the vano model

    with simple MLP ecnoder and decoder

    going to condition on artery one hot encoding

    last layer of encoder == 2* latent dim
    """

    def __init__(
        self,
        layers_encoder: List[int],
        layers_decoder: List[int],
        latent_dim: int,
    ):
        super(VANO, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = self.build_encoder(layers_encoder)
        self.decoder = self.build_decoder(layers_decoder)

    def build_encoder(self, layers: List[int]):
        encoder = nn.ModuleList()
        for i in range(len(layers) - 1):
            encoder.append(nn.Linear(layers[i], layers[i + 1]))
        return encoder

    def build_decoder(self, layers: List[int]):
        decoder = nn.ModuleList()
        for i in range(
            len(layers) - 1,
        ):
            decoder.append(nn.Linear(layers[i], layers[i + 1]))
        return decoder

    def encode(self, x, condition):
        x = torch.cat((x, condition), -1)
        for layer in self.encoder:
            x = layer(x)
            x = F.gelu(x)
        mean = x[:, : self.latent_dim]
        log_var = x[:, self.latent_dim :]
        return mean, log_var

    def decode(self, x, condition):
        x = torch.cat((x, condition), -1)
        for layer in self.decoder[:-1]:
            x = F.gelu(layer(x))
        return self.decoder[-1](x)

    def sample_gauss(self, mean, log_var):
        epsilon = torch.randn_like(mean)
        return mean + torch.exp(log_var / 2) * epsilon

    def forward(self, x, condition):
        mean, log_var = self.encode(x, condition)
        z = self.sample_gauss(mean, log_var)

        pred = self.decode(z, condition)
        return mean, log_var, z, pred
