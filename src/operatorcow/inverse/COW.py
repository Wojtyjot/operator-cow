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
# TODO device checks everywhere


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
        self.mesurement = mesurement
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
            self.mesurement_value = self.g.ndata["y"][
                100 * 100 : 100 * 101, 2
            ].squeeze()
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

            return self.g, torch.cat((self.theta, self.SV)), in_f

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

    def set_p_out(self, p_out: torch.Tensor):
        self.p_out = p_out

    def get_p_out(self):
        return self.p_out

    def set_SV(self, SV: torch.Tensor):
        self.SV = SV

    def has_mesurement(self):
        return self.mesurement


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
        self.rho = 1.06  ## must be in CGS units
        self.SV_true = None
        self.SV = (
            torch.Tensor(np.random.uniform(70, 140)).to(device).requires_grad_(True)
        )
        self.device = device
        self.model_surrogate = model_surrogate
        self.AE_model = AE_model
        self.track = track
        self.normalizer_x = normalizer_x
        self.normalizer_y = normalizer_y
        self.normalizer_theta = normalizer_theta
        self.joints_path = joints_path  # TODO
        self.load_data(data_path)
        self.create_optimizer()
        self.propagate_SV()

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
            g.ndata["y"] = torch.from_numpy(Y).float()  # y można wyjebać w większości
            # y potrzebny tlko do "pomiarów" bedzie lepiej pamięciowo
            # trzeba pamietac by usunac SV i dodawac z cow
            theta = torch.from_numpy(theta).float()
            # passing also true values for comparison
            input_f = [torch.from_numpy(in_func).float() for in_func in in_funcs]
            # moze na poczatek zrobic hardcoded mesurements?
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
            elif artery in [
                "L_MCA",
                "R_MCA",
                "L_ACA_A1",
                "R_ACA_A1",
                "L_PCA_P1",
                "R_PCA_P1",
                "L_ACA_A2",
                "R_ACA_A2",
                "L_PCA_P2",
                "R_PCA_P2",
            ]:
                self.arteries.append(
                    Artery(
                        g,
                        input_f,
                        theta[:-1],
                        name=artery,
                        device=self.device,
                        mesurement=True,
                    )
                )

            else:
                self.arteries.append(
                    Artery(g, input_f, theta[:-1], name=artery, device=self.device)
                )  # Theta in arteries is without SV

    def create_joints(self, joints_csv):
        """
        Function reads topography info from joints csv file
        every joint is represented as tuple (p, d1, d2, merging)
        where p is parent artery, d1 and d2 are daughter arteries
        if merging is True then 2 daughter arteries are merged into 1 parent artery
        At every joint bifurcation loss is calculated

        Args:
            joints_csv (str): path to csv file with joints properties

        joints_csv file must have following columns:
        id(int): numeric id of joint
        p(int): id of parent artery
        d1(int): id of daughter artery 1
        d2(int): id of daughter artery 2
        merging(bool): indicator function if merging is present
        """
        joints = list()
        jdf = pd.read_csv(joints_csv)
        for id in jdf.id:
            joints.append(
                (
                    str(jdf[jdf.id == id].p.iloc[0]),
                    str(jdf[jdf.id == id].d1.iloc[0]),
                    str(jdf[jdf.id == id].d2.iloc[0]),
                    (jdf[jdf.id == id].merging == 1).bool(),
                )
            )
        return joints

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

            yield batched

    def solve_arteries(self, batch_size: int):
        """
        Function that computes solution for single artery....
        Lepiej bachowć
        """
        for batch, idx in self.loader_GNOT(batch_size):
            g, u_p, g_u = batch  # znormalizowac trzeba to

            g.ndata["x"] = self.normalizer_x.transform(g.ndata["x"], inverse=False)
            u_p = self.normalizer_theta.transform(u_p, inverse=False)

            g, u_p, g_u = g.to(self.device), u_p.to(self.device), g_u.to(self.device)

            out = self.model_surrogate(
                g, u_p, g_u
            )  # trzeba zrobic reshape bo jest [bs * n_nodes, 3]
            out = self.normalizer_y.transform(out, inverse=True)
            out = out.reshape(batch_size, -1, 3)  # mam nadzieje ze to dobrze

            # tu musi byc funkcja do zapisu wynikow do artery
            self.update_arteries(out, idx)

    def get_reg_loss(self, batch_size: int):  # raczej różny bs dla AE i GNOT
        """
        Function computes regularization loss from AE
        """
        reg_loss = 0
        for batch in self.loader_AE(batch_size):
            u_bc = batch[0]
            u_bc = self.normalizer_x.transform(u_bc, inverse=False)
            u_bc = u_bc.to(self.device)
            out = self.AE_model(u_bc)
            out = self.normalizer_y.transform(out, inverse=True)
            out = out.reshape(batch_size, -1, 1)
            reg_loss += nn.MSELoss()(u_bc, out)
        return reg_loss

    def compute_bifurcation_loss(self):
        """
        Function computes bifurcation loss

        conservation of mass and total pressure cont
        """
        loss_mass = 0
        loss_pressure = 0
        for joint in self.joints:
            p, d1, d2, merging = joint
            p = self.arteries[p]
            d1 = self.arteries[d1]
            d2 = self.arteries[d2]
            if merging:
                loss_mass += torch.mean(
                    torch.square(
                        p.get_u_in() * p.get_a_in()
                        - d1.get_u_out() * d1.get_a_out()
                        - d2.get_u_out() * d2.get_a_out()
                    )
                )
                loss_pressure += torch.mean(
                    torch.square(
                        p.get_p_in()
                        + 0.5 * self.rho * torch.square(p.get_u_in())
                        - d1.get_p_out()
                        - 0.5 * self.rho * torch.square(d1.get_u_out())
                    )
                )
                loss_pressure += torch.mean(
                    torch.square(
                        p.get_p_in()
                        + 0.5 * self.rho * torch.square(p.get_u_in())
                        - d2.get_p_out()
                        - 0.5 * self.rho * torch.square(d2.get_u_out())
                    )
                )
            else:
                loss_mass += torch.mean(
                    torch.square(
                        p.get_u_out() * p.get_a_out()
                        - d1.get_u_in() * d1.get_a_in()
                        - d2.get_u_in() * d2.get_a_in()
                    )
                )
                loss_pressure += torch.mean(
                    torch.square(
                        p.get_p_out()
                        + 0.5 * self.rho * torch.square(p.get_u_out())
                        - d1.get_p_in()
                        - 0.5 * self.rho * torch.square(d1.get_u_in())
                    )
                )
                loss_pressure += torch.mean(
                    torch.square(
                        p.get_p_out()
                        + 0.5 * self.rho * torch.square(p.get_u_out())
                        - d2.get_p_in()
                        - 0.5 * self.rho * torch.square(d2.get_u_in())
                    )
                )
        return loss_mass, loss_pressure

    def compute_mesurement_loss(self):
        """
        Function computes mesurement loss
        """
        raise NotImplementedError

    def compute_SV_loss(self):
        """
        Function computes stroke volume regularization loss
        """
        raise NotImplementedError

    def update_arteries(self, pred: torch.Tensor, idx: List[int]):
        """
        Function to update arteries with new predictions

        pred from = [
            [[p(x0, t0), a(x0, t0), u(x0, t0)]
            [p(x0, t1), a(x0, t1), u(x0, t1)]
        ]
        """

        for i in idx:
            self.arteries[i].set_u_in(pred[i, :100, 2])
            self.arteries[i].set_u_out(pred[i, -100:, 2])
            self.arteries[i].set_a_in(pred[i, :100, 1])
            self.arteries[i].set_a_out(pred[i, -100:, 1])
            self.arteries[i].set_p_in(pred[i, :100, 0])
            self.arteries[i].set_p_out(pred[i, -100:, 0])

    def get_arteries(self, idx, model: str = "GNOT"):
        try:
            return self.arteries[idx].get_inputs(model)
        except ValueError:
            pass

    def propagate_SV(self):
        """
        Function passes SV value to arteries
        """
        for artery in self.arteries:
            artery.set_SV(self.SV)

    def solve(self):
        """
        Function for solving inverse problem on whole COW
        """

        raise NotImplementedError
