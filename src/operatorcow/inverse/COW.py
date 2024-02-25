from pathlib import Path
from typing import *

import dgl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import wandb
from utils.utils import MultipleTensors

## Wszystko musi byc na same device
# artery musi zwracać wszystkie dane potrzebne do obliczen
# g, inputs_f
# częśc theta tj. Ls i r0s bedzie atrybutem COW
# INPUTY NIE SĄ ZNORMALIZOWANE !!!!!
# ZNORMALIZOWAC PRZED WRZUCENIEM DO MODELU
# NORMALIZACJA RACZEJ W SCOPE COW
# A0 tez do lossa dodac


class Artery(object):
    """
    Class for representing artery in model

    stores data and parameters associated with artery
    """

    def __init__(
        self,
        g: dgl.DGLGraph,
        inputs_f: List[torch.Tensor],
        theta: torch.Tensor,
        name: str,
        device: str,
    ):
        self.name = name
        self.g = g
        self.device = device
        self.theta = theta
        self.inputs_f = inputs_f
        self.L = g.ndata["x"][-1, 0]
        self.initialize_parameters(inputs_f, theta)

    def initialize_parameters(self, mesurement: bool):
        """
        Function to initialize parameters to optimize

        for now mesurement is assumed to be in a middle of artery
        """
        parameters = []

        if mesurement:
            u_bc = (
                self.g.ndata["y"][100 * 100 : 100 * 101, 2]
                .squeeze()
                .to(self.device)
                .rquires_grad_(True)
            )
        else:
            u_bc = torch.rand(100, 1).to(self.device).requires_grad_(True)

        u0 = u_bc[0].detach().repeat(200, 1).to(self.device).requires_grad_(True)
        parameters.append(u_bc)
        parameters.append(u0)

        self.parameters = parameters

    def parameters(self):
        """
        Function return parameters to optimize

        """
        return self.parameters

    def get_u_BC(self):
        """
        Function return boundary condition
        """
        t = torch.linspace(0, 1, 100).to(self.device).reshape(-1, 1)
        x = torch.zeros(100, 1).to(self.device).reshape(-1, 1)
        return torch.cat((x, t, self.parameters[0].reshape(-1, 1)), dim=-1)

    def get_u0(self):
        """
        Function returns u0 initial condition
        """
        x = torch.linspace(0, self.L, 200).to(self.device).reshape(-1, 1)
        t = torch.zeros(200, 1).to(self.device).reshape(-1, 1)
        return torch.cat((x, t, self.parameters[1]), dim=-1)

    def get_inputs(self, model: str):
        """
        Function return inputs to model
        """
        if model == "GNOT":
            p0, a0 = self.inputs_f[1], self.inputs_f[3]
            u0 = self.get_u0()
            in_bc = self.get_u_BC()
            in_f = MultipleTensors([i for i in (in_bc, p0, u0, a0)])

            return self.g, self.theta, in_f

        elif model == "AE":
            return self.parameters[0]

        else:
            raise ValueError("Model not recognized must be GNOT or AE")


class COW(object):
    """
    Class for representing circle of willis model
    and solving inverse problem asociated with it
    TODO add modr description
    """

    # cow musi posiadac ogolne parametry

    def __init__(
        self,
        model_surrogate: nn.Module,
        AE_model: nn.Module,
        data_path: str,
        track: bool,
        device: str,
        normalizer_x,
        normalizer_y,
        normalizer_theta,
    ):
        self.load_data(data_path)

    def load_data(self, data_path: str):
        """
        Function to load data

        Creates artery objecs ...

        Will read pre specified npy files for testing

        and transform them into usable format

        Parameters
        ----------
        data_path: str
            path to folder containing npy files with data
        """
        arteries = [
            "L_int_carotid_I",
            "R_int_carotid_I",
            # "R_vertebral",
            # "L_vertebral",
            "Basilar",
            "L_int_carotid_II",
            "R_int_carotid_II",
            "L_PcoA",
            "R_PcoA",
            "L_MCA",
            "R_MCA",
            "L_ACA_A1",
            "R_ACA_A1",
            "L_PCA_P1",
            "R_PCA_P1",
            "L_ACA_A2",
            "R_ACA_A2",
            "AcoA",
            "L_PCA_P2",
            "R_PCA_P2",
        ]
        self.arteries = []

        for artery in arteries:
            X, Y, theta, in_funcs = np.load(
                data_path + artery + ".npy", allow_pickle=True
            )
            g = dgl.DGLGraph()
            g.add_nodes(X.shape[0])
            g.ndata["x"] = torch.from_numpy(X).float()
            g.ndata["y"] = torch.from_numpy(Y).float()
            # trzeba pamietac by usunac SV i dodawac z cow
            theta = torch.from_numpy(theta).float()
            # passing also true values for comparison
            input_f = [torch.from_numpy(in_func).float() for in_func in in_funcs]

            self.arteries.append(
                Artery(g, input_f, theta, name=artery, device=self.device)
            )
