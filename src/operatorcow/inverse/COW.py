from pathlib import Path
from typing import *

import dgl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import wandb
from torch.nn.utils.rnn import pad_sequence
from utils.utils import MultipleTensors

## Wszystko musi byc na same device
# artery musi zwracać wszystkie dane potrzebne do obliczen
# g, inputs_f
# częśc theta tj. Ls i r0s bedzie atrybutem COW
# INPUTY NIE SĄ ZNORMALIZOWANE !!!!!
# ZNORMALIZOWAC PRZED WRZUCENIEM DO MODELU
# NORMALIZACJA RACZEJ W SCOPE COW
# A0 tez do lossa dodac
# Dodac jeszcze ze jezeli root to bc bez parmetrow


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
        mesurement: bool = False,
        root: bool = False,
    ):
        self.name = name
        self.g = g
        self.device = device
        self.theta = theta
        self.inputs_f = inputs_f
        self.L = g.ndata["x"][-1, 0]
        self.root = root
        self.initialize_parameters(mesurement)

    def initialize_parameters(self, measurement: bool):
        """
        Function to initialize parameters to optimize

        for now, measurement is assumed to be in the middle of the artery
        """
        if measurement:
            self.u_bc = (
                self.g.ndata["y"][100 * 100 : 100 * 101, 2]
                .squeeze()
                .to(self.device)
                .requires_grad_(True)
            )
            u0 = self.u_bc[0].detach().repeat(200, 1).requires_grad_(True)
            self.parameters = [self.u_bc, u0]

        elif self.root:
            self.u_bc = (
                self.g.ndata["y"][0:100, 2]
                .squeeze()
                .to(self.device)
                .requires_grad_(True)
            )
            u0 = self.u_bc[0].detach().repeat(200, 1).requires_grad_(True)
            self.parameters = [u0]

        else:
            u_bc = torch.rand(100, 1).to(self.device).requires_grad_(True)
            u0 = u_bc[0].detach().repeat(200, 1).requires_grad_(True)
            self.parameters = [u_bc, u0]

    def get_parameters(self):
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

        if self.root:
            return torch.cat((x, t, self.u_bc.reshape(-1, 1)), dim=-1)
        else:
            return torch.cat((x, t, self.parameters[0].reshape(-1, 1)), dim=-1)

    def get_u0(self):
        """
        Function returns u0 initial condition
        """
        x = torch.linspace(0, self.L, 200).to(self.device).reshape(-1, 1)
        t = torch.zeros(200, 1).to(self.device).reshape(-1, 1)

        if self.root:
            return torch.cat((x, t, self.parameters[0]), dim=-1)
        else:
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
            if self.root:
                raise ValueError("Root artery not supported for AE model")

            return self.parameters[0].reshape(1, 100)

        else:
            raise ValueError("Model not recognized must be GNOT or AE")

    def set_u_in(self, u_in: torch.Tensor):
        self.u_in = u_in

    def get_u_in(self):
        return self.u_in

    def set_u_out(self, u_out: torch.Tensor):
        self.u_out = u_out

    def get_u_out(self):
        return self.u_out

    def set_a_in(self, a_in: torch.Tensor):
        self.a_in = a_in

    def get_a_in(self):
        return self.a_in

    def set_a_out(self, a_out: torch.Tensor):
        self.a_out = a_out

    def get_a_out(self):
        return self.a_out

    def set_p_in(self, p_in: torch.Tensor):
        self.p_in = p_in

    def get_p_in(self):
        return self.p_in


class COW(object):
    """
    Class for representing circle of willis model
    and solving inverse problem asociated with it
    TODO add modr description
    """

    # cow musi posiadac ogolne parametry
    # trzeba jakos dodac sv do theta naczyn

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
        joints_path: str,
    ):
        self.SV_true = None
        self.SV = (
            torch.Tensor(np.random.uniform(70, 140)).to(device).requires_grad_(True)
        )
        self.device = device
        self.load_data(data_path)
        self.create_optimizer()

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
            if self.SV_true is None:
                self.SV_true = theta[-1]
            g = dgl.DGLGraph()
            g.add_nodes(X.shape[0])
            g.ndata["x"] = torch.from_numpy(X).float()
            g.ndata["y"] = torch.from_numpy(Y).float()
            # trzeba pamietac by usunac SV i dodawac z cow
            theta = torch.from_numpy(theta).float()
            # passing also true values for comparison
            input_f = [torch.from_numpy(in_func).float() for in_func in in_funcs]
            if artery in ["L_int_carotid_I", "R_int_carotid_I", "Basilar"]:
                self.arteries.append(
                    Artery(
                        g,
                        input_f,
                        theta[:-1],
                        name=artery,
                        device=self.device,
                        root=True,
                    )
                )
            else:
                self.arteries.append(
                    Artery(g, input_f, theta[:-1], name=artery, device=self.device)
                )  # Theta in arteries is without SV

    def create_optimizer(self, lr: float = 0.5):
        """
        Function creates a single optimizer for all arterial parameters
        """
        # TODO test z LBFGS tylko colsure trzeba zdefiniowac
        p = []
        for artery in self.arteries:
            p.extend(artery.get_parameters())
        self.optimizer = torch.optim.Adam(p, lr=lr)

    def loader_GONT(self, batch_size):
        # Loader dla gnota req loss mozna w jednym batchu?
        batch_idx = [
            list(range(i, min(i + batch_size, len(self.arteries))))
            for i in range(0, len(self.arteries), batch_size)
        ]
        for indices in batch_idx:
            transposed = zip(*[self.get_arteries(idx, "GNOT") for idx in indices])
            batched = []
            for sample in transposed:
                if isinstance(sample[0], dgl.DGLGraph):
                    batched.append(dgl.batch(list(sample)))
                elif isinstance(sample[0], torch.Tensor):
                    batched.append(torch.stack(sample))
                elif isinstance(sample[0], MultipleTensors):
                    sample_ = MultipleTensors(
                        [
                            pad_sequence(
                                [sample[i][j] for i in range(len(sample))]
                            ).permute(1, 0, 2)
                            for j in range(len(sample[0]))
                        ]
                    )
                    batched.append(sample_)
                else:
                    raise NotImplementedError
            yield batched, indices

    def loader_AE(self, batch_size):
        # Loader dla AE # yield also indicies
        batch_idx = [
            list(range(i, min(i + batch_size, len(self.arteries))))
            for i in range(3, len(self.arteries), batch_size)  # idx 0,1,2 to root
        ]
        for indices in batch_idx:
            transposed = zip(*[self.get_arteries(idx, "AE") for idx in indices])
            batched = []
            for sample in transposed:
                if isinstance(sample[0], torch.Tensor):
                    batched.append(torch.stack(sample))
                else:
                    raise NotImplementedError

            yield batched, indices

    def solve_arteries(self, batch_size: int):
        """
        Function that computes solution for single artery....
        Lepiej bachowć
        """
        for batch, idx in self.loader_GNOT(batch_size):
            g, u_p, g_u = batch

            g, u_p, g_u = g.to(self.device), u_p.to(self.device), g_u.to(self.device)

            out = self.model_surrogate(
                g, u_p, g_u
            )  # trzeba zrobic reshape bo jest [bs * n_nodes, 3]
            out = out.reshape(batch_size, -1, 3)  # mam nadzieje ze to dobrze

    def get_arteries(self, idx, model: str = "GNOT"):
        try:
            return self.arteries[idx].get_inputs(model)
        except ValueError:
            pass
