from pathlib import Path
from typing import *

import dgl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import wandb
from data_utils import WeightedLpRelLoss
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
        VANO: bool = True,
        normalizer_u_bc=None,
    ):
        self.name = name
        self.g = g
        self.device = device
        self.theta = theta
        self.inputs_f = inputs_f
        self.L = g.ndata["x"][-1, 0]
        self.root = root
        self.mesurement = mesurement
        self.VANO = VANO
        if normalizer_u_bc is not None:
            self.normalizer_u_bc = normalizer_u_bc
        self.initialize_parameters(mesurement)

        # del self.g.ndata['y']

    def initialize_parameters(self, measurement: bool):
        """
        Function to initialize parameters to optimize

        for now, measurement is assumed to be in the middle of the artery
        """
        if self.VANO:
            if measurement:

                self.u_bc_latent = (
                    torch.randn(1, 16).to(self.device).requires_grad_(True)
                )
                # print(self.u_bc.shape)
                # u0 = self.u_bc[0].detach().repeat(200, 1).requires_grad_(True)
                self.mesurement_value = (
                    self.g.ndata["y"][100 * 100 : 100 * 101, 2]
                    .squeeze()
                    .to(self.device)
                )
                self.parameters = [self.u_bc_latent]
                self.u_bc_true = self.g.ndata["y"][:100, 2].squeeze()

            elif self.root:
                self.u_bc = (
                    self.g.ndata["y"][0:100, 2]
                    .unsqueeze(-1)
                    .to(self.device)
                    .requires_grad_(True)
                )
                # u0 = self.u_bc[0].detach().repeat(200, 1).requires_grad_(True)
                # self.parameters = [u0]
                self.u_bc_true = self.g.ndata["y"][:100, 2].squeeze()
                self.parameters = None

            else:
                self.u_bc_latent = (
                    torch.randn(1, 16).to(self.device).requires_grad_(True)
                )
                # u0 = self.u_bc[0].detach().repeat(200, 1).requires_grad_(True)
                self.parameters = [self.u_bc_latent]
                self.u_bc_true = self.g.ndata["y"][:100, 2].squeeze()

        else:

            if measurement:
                self.u_bc = (
                    self.g.ndata["y"][100 * 100 : 100 * 101, 2]
                    .unsqueeze(-1)
                    .to(self.device)
                    .requires_grad_(True)
                )
                print(self.u_bc.shape)
                u0 = self.u_bc[0].detach().repeat(200, 1).requires_grad_(True)
                self.mesurement_value = (
                    self.g.ndata["y"][100 * 100 : 100 * 101, 2]
                    .squeeze()
                    .to(self.device)
                )
                self.parameters = [self.u_bc, u0]
                self.u_bc_true = self.g.ndata["y"][:100, 2].squeeze()

            elif self.root:
                self.u_bc = (
                    self.g.ndata["y"][0:100, 2]
                    .unsqueeze(-1)
                    .to(self.device)
                    .requires_grad_(True)
                )
                u0 = self.u_bc[0].detach().repeat(200, 1).requires_grad_(True)
                self.parameters = [u0]
                self.u_bc_true = self.g.ndata["y"][:100, 2].squeeze()

            else:
                self.u_bc = torch.rand(100, 1).to(self.device).requires_grad_(True)
                u0 = self.u_bc[0].detach().repeat(200, 1).requires_grad_(True)
                self.parameters = [self.u_bc, u0]
                self.u_bc_true = self.g.ndata["y"][:100, 2].squeeze()

    def get_parameters(self):
        """
        Function return parameters to optimize

        """
        return self.parameters

    def get_u_BC(self, VANO_model: nn.Module = None, normalizer_u_bc=None):
        """
        Function return boundary condition
        """
        t = torch.linspace(0, 1, 100).to(self.device).reshape(-1, 1)
        x = torch.zeros(100, 1).to(self.device).reshape(-1, 1)
        if self.VANO:
            if self.root:
                return torch.cat((x, t, self.u_bc.reshape(-1, 1)), dim=-1)
            else:
                # need to doceode u_bc and then renormalize
                u_bc = VANO_model.decode(
                    self.parameters[0].reshape(1, 16), self.theta[:11].reshape(1, 11)
                )
                u_bc = normalizer_u_bc.transform(u_bc, inverse=True)
                return torch.cat((x, t, u_bc.reshape(-1, 1)), dim=-1)
                # return torch.cat((x, t, self.parameters[0].reshape(-1, 1)), dim=-1)

        else:
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

    def get_inputs(
        self, model: str, VANO_model: nn.Module = None, normalizer_u_bc=None
    ):
        """
        Function return inputs to model
        """
        if model == "GNOT":
            p0, u0, a0 = self.inputs_f[1], self.inputs[2], self.inputs_f[3]
            # u0 = self.get_u0()
            in_bc = self.get_u_BC(
                VANO_model=VANO_model, normalizer_u_bc=normalizer_u_bc
            )
            in_f = MultipleTensors([i for i in (in_bc, p0, u0, a0)])

            return self.g, torch.cat((self.theta.to(self.device), self.SV)), in_f

        elif model == "AE":
            if self.root:
                raise ValueError("Root artery not supported for AE model")

            return self.parameters[0].reshape(1, 100)

        else:
            raise ValueError("Model not recognized must be GNOT or AE")

    def log(self):
        """
        Function logs to wandb the artery parameters
        u_bc as plot
        """

        plt.plot(self.u_bc.detach().cpu().numpy(), label="reconstructed")
        plt.plot(self.u_bc_true.detach().cpu().numpy(), label="true")
        plt.title(f"U_bc for artery {self.name}")
        plt.legend()
        wandb.log({self.name: plt})
        plt.close()

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

    def set_reconstructed_u_mesurement(self, u):  # trzeba zmienić nazwy zmiennych
        self.u_mesurement = u

    def get_reconstructed_u_mesurement(self):
        return self.u_mesurement

    def get_true_mesurement(self):
        return self.mesurement_value


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
        lr: float,
        VANO: bool,
        model_VANO: nn.Module,
        normalizer_u_bc=None,
    ):
        self.rho = 1.06  ## must be in CGS units
        self.SV_true = None
        self.SV = (
            torch.Tensor([np.random.uniform(70, 140)]).to(device).requires_grad_(True)
        )
        self.VANO = VANO
        self.device = device
        self.model_surrogate = model_surrogate
        self.AE_model = AE_model
        self.model_VANO = model_VANO
        self.track = track
        self.normalizer_x = normalizer_x
        self.normalizer_y = normalizer_y
        if normalizer_u_bc is not None:
            self.normalizer_u_bc = normalizer_u_bc
        self.normalizer_theta = normalizer_theta
        self.joints_path = joints_path  # TODO
        self.l2_loss = WeightedLpRelLoss(p=2, component="all", normalizer=None)
        self.load_data(data_path)
        self.create_optimizer(lr)
        self.propagate_SV()
        self.joints = self.create_joints(joints_path)

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
                        mesurement=False,
                        VANO=self.VANO,
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
                        VANO=self.VANO,
                    )
                )

            else:
                self.arteries.append(
                    Artery(
                        g,
                        input_f,
                        theta[:-1],
                        name=artery,
                        device=self.device,
                        mesurement=False,
                        VANO=self.VANO,
                    )
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
            if artery.get_parameters() is not None:
                p.extend(artery.get_parameters())
        # print(self.SV.requires_grad_(True).is_leaf)
        p.extend([self.SV.requires_grad_(True)])
        self.optimizer = torch.optim.Adam(p, lr=lr)

    def loader_GNOT(self, batch_size, batch_idx=None):
        # Loader dla gnota req loss mozna w jednym batchu?
        if batch_idx is not None:
            batch_idx = [batch_idx]
        else:
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

    def loader_AE(self, batch_size, batch_idx=None):
        # Loader dla AE # yield also indicies
        if 0 in batch_idx:
            batch_idx.remove(0)
        elif 1 in batch_idx:
            batch_idx.remove(1)
        elif 2 in batch_idx:
            batch_idx.remove(2)

        if batch_idx is not None:
            batch_idx = [batch_idx]
        else:
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

    def solve_arteries(self, batch_size: int, batch_idx=None):
        """
        Function that computes solution for single artery....
        Lepiej bachowć
        """
        # with torch.no_grad(): # może rozjebać wszystko ale zobaczmy
        if batch_idx is not None:
            batch_idx = batch_idx
            # TODO
            for batch, idx in self.loader_GNOT(batch_size, batch_idx):
                g, u_p, g_u = batch

                g, u_p, g_u = (
                    g.to(self.device),
                    u_p.to(self.device),
                    g_u.to(self.device),
                )

                g.ndata["x"] = self.normalizer_x.transform(g.ndata["x"], inverse=False)
                u_p = self.normalizer_theta.transform(u_p, inverse=False)

                # g, u_p, g_u = g.to(self.device), u_p.to(self.device), g_u.to(self.device)

                out = self.model_surrogate(
                    g, u_p, g_u
                )  # trzeba zrobic reshape bo jest [bs * n_nodes, 3]
                out = self.normalizer_y.transform(out, inverse=True)
                out = out.reshape(len(batch_idx), -1, 3)  # mam nadzieje ze to dobrze

                # tu musi byc funkcja do zapisu wynikow do artery
                # print(idx)
                self.update_arteries(out, idx)

        else:
            for batch, idx in self.loader_GNOT(batch_size):

                g, u_p, g_u = batch  # znormalizowac trzeba to

                g, u_p, g_u = (
                    g.to(self.device),
                    u_p.to(self.device),
                    g_u.to(self.device),
                )

                g.ndata["x"] = self.normalizer_x.transform(g.ndata["x"], inverse=False)
                u_p = self.normalizer_theta.transform(u_p, inverse=False)

                # g, u_p, g_u = g.to(self.device), u_p.to(self.device), g_u.to(self.device)

                out = self.model_surrogate(
                    g, u_p, g_u
                )  # trzeba zrobic reshape bo jest [bs * n_nodes, 3]
                out = self.normalizer_y.transform(out, inverse=True)
                out = out.reshape(batch_size, -1, 3)  # mam nadzieje ze to dobrze

                # tu musi byc funkcja do zapisu wynikow do artery
                # print(idx)
                self.update_arteries(out, idx)

    def compute_validation_l2_loss(self, batch_size: int):
        """
        Function computes validation loss for all arteries
        """
        loss = 0
        i = 0
        for batch, idx in self.loader_GNOT(batch_size):
            i += 1
            g, u_p, g_u = batch

            g.ndata["x"] = self.normalizer_x.transform(g.ndata["x"], inverse=False)
            u_p = self.normalizer_theta.transform(u_p, inverse=False)

            g, u_p, g_u = g.to(self.device), u_p.to(self.device), g_u.to(self.device)

            out = self.model_surrogate(g, u_p, g_u)
            out = self.normalizer_y.transform(out, inverse=True)
            y_true = g.ndata["y"].squeeze().to(self.device)
            loss += self.l2_loss(g, out, y_true)
        return loss / i

    def get_reg_loss(
        self, batch_size: int, batch_idx=None
    ):  # raczej różny bs dla AE i GNOT
        """
        Function computes regularization loss from AE
        """
        reg_loss = 0

        if batch_idx is not None:
            batch_idx = batch_idx
            for batch in self.loader_AE(batch_size, batch_idx):
                u_bc = batch[0]
                # normalization needs to be done  ned to add dimensions 2
                u_bc = torch.cat(
                    (
                        torch.zeros(batch[0].shape[0], 100, 2).to(self.device),
                        u_bc.unsqueeze(-1),
                    ),
                    dim=-1,
                )
                u_bc = u_bc.reshape(-1, 3)
                u_bc = self.normalizer_y.transform(u_bc, inverse=False)
                u_bc = u_bc.reshape(batch[0].shape[0], -1, 3)
                u_bc = u_bc[:, :, -1].squeeze()
                u_bc = u_bc.to(self.device)
                out = self.AE_model(u_bc)
                try:
                    out = torch.cat(
                        (
                            torch.zeros(batch[0].shape[0], 100, 2).to(self.device),
                            out.unsqueeze(-1),
                        ),
                        dim=-1,
                    )
                except:
                    out = torch.cat(
                        (torch.zeros(100, 2).to(self.device), out.unsqueeze(-1)), dim=-1
                    )
                out = out.reshape(-1, 3)
                out = self.normalizer_y.transform(out, inverse=True)
                out = out.reshape(batch[0].shape[0], -1, 3)
                out = out[:, :, -1].squeeze()
                reg_loss += nn.MSELoss()(u_bc, out)
        else:
            for batch in self.loader_AE(batch_size):
                u_bc = batch[0]
                # normalization needs to be done  ned to add dimensions 2
                # print(f"u_bc.shape = {u_bc.shape}")
                u_bc = torch.cat(
                    (
                        torch.zeros(batch[0].shape[0], 100, 2).to(self.device),
                        u_bc.unsqueeze(-1),
                    ),
                    dim=-1,
                )
                # print(f"u_bc.shape = {u_bc.shape}")
                u_bc = u_bc.reshape(-1, 3)
                u_bc = self.normalizer_y.transform(u_bc, inverse=False)
                u_bc = u_bc.reshape(batch[0].shape[0], -1, 3)
                u_bc = u_bc[:, :, -1].squeeze()
                u_bc = u_bc.to(self.device)
                # print(f"u_bc.shape before me = {u_bc.shape}")
                out = self.AE_model(u_bc)
                # print(out.shape)
                try:
                    out = torch.cat(
                        (
                            torch.zeros(batch[0].shape[0], 100, 2).to(self.device),
                            out.unsqueeze(-1),
                        ),
                        dim=-1,
                    )
                except:
                    out = torch.cat(
                        (torch.zeros(100, 2).to(self.device), out.unsqueeze(-1)), dim=-1
                    )
                out = out.reshape(-1, 3)
                out = self.normalizer_y.transform(out, inverse=True)
                out = out.reshape(batch[0].shape[0], -1, 3)
                out = out[:, :, -1].squeeze()
                reg_loss += nn.MSELoss()(u_bc, out)
        return reg_loss

    def compute_bifurcation_loss(self, j=None):
        """
        Function computes bifurcation loss

        conservation of mass and total pressure cont
        """
        loss_mass = 0
        loss_pressure = 0
        if j is not None:
            p, d1, d2, merging = j
            p = self.arteries[int(p)]
            d1 = self.arteries[int(d1)]
            d2 = self.arteries[int(d2)]
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

        else:
            for joint in self.joints:
                p, d1, d2, merging = joint
                p = self.arteries[int(p)]
                d1 = self.arteries[int(d1)]
                d2 = self.arteries[int(d2)]
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

    def compute_mesurement_loss(self, batch_idx=None):
        """
        Function computes mesurement loss
        """
        loss = 0
        if batch_idx is not None:
            for idx in batch_idx:
                if self.arteries[idx].has_mesurement():
                    loss += nn.MSELoss()(
                        self.arteries[idx].get_reconstructed_u_mesurement(),
                        self.arteries[idx].get_true_mesurement(),
                    )
        else:
            for artery in self.arteries:
                if artery.has_mesurement():
                    loss += nn.MSELoss()(
                        artery.get_reconstructed_u_mesurement(),
                        artery.get_true_mesurement(),
                    )
        return loss

    def compute_SV_loss(self):
        """
        Function computes stroke volume regularization loss
        """
        if self.SV < 70:
            return torch.square(self.SV - 70)
        elif self.SV > 140:
            return torch.square(self.SV - 140)
        else:
            return 0

    def update_arteries(self, pred: torch.Tensor, idx: List[int]):
        """
        Function to update arteries with new predictions

        pred from = [
            [[p(x0, t0), a(x0, t0), u(x0, t0)]
            [p(x0, t1), a(x0, t1), u(x0, t1)]
        ]
        """

        for i, idx in enumerate(idx):
            # print(pred.shape)
            # print(f"i == {i}")
            # print(f"idx = {idx}")
            self.arteries[idx].set_u_in(pred[i, :100, 2])
            self.arteries[idx].set_u_out(pred[i, -100:, 2])
            self.arteries[idx].set_a_in(pred[i, :100, 1])
            self.arteries[idx].set_a_out(pred[i, -100:, 1])
            self.arteries[idx].set_p_in(pred[i, :100, 0])
            self.arteries[idx].set_p_out(pred[i, -100:, 0])
            if self.arteries[idx].has_mesurement():
                self.arteries[idx].set_reconstructed_u_mesurement(
                    pred[i, 100 * 100 : 100 * 101, 2]
                )

    def get_arteries(self, idx, model: str = "GNOT"):
        try:
            return self.arteries[idx].get_inputs(
                model, self.model_VANO, self.normalizer_u_bc
            )
        except ValueError:
            pass

    def propagate_SV(self):
        """
        Function passes SV value to arteries
        """
        for artery in self.arteries:
            artery.set_SV(self.SV)

    def log_arteries(self):
        """
        Function logs all arteries to wandb
        """
        for artery in self.arteries:
            artery.log()

    def solve(
        self,
        max_iters: int,
        eps: float,
        batch_size: int,
        lambda_reg: float,
        lambda_mes: float,
        lambda_sv: float,
        lambda_bif: float,
        log_every: int,
    ):
        """
        Function for solving inverse problem on whole COW:

        step 1 solve each artery separately
        step 2 ccompute loss,
        step 3 compute gradients
        step 4 update trainable parameters
        """
        iter = 0
        for i in range(max_iters):
            iter += 1
            self.solve_arteries(batch_size)
            loss = 0
            loss += lambda_reg * self.get_reg_loss(batch_size)
            loss += lambda_mes * self.compute_mesurement_loss()
            loss += lambda_sv * self.compute_SV_loss()
            loss_mass, loss_pressure = self.compute_bifurcation_loss()
            loss += lambda_bif * (loss_mass + loss_pressure)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.propagate_SV()
            print(loss.item())
            if self.track and iter % log_every == 0:
                wandb.log(
                    {
                        "loss": loss.item(),
                        "reg_loss": lambda_reg * self.get_reg_loss(batch_size).item(),
                        "mes_loss": lambda_mes * self.compute_mesurement_loss().item(),
                        "SV_loss": lambda_sv * self.compute_SV_loss(),
                        "bif_loss": lambda_bif * (loss_mass + loss_pressure).item(),
                    }
                )
                self.log_arteries()
                wandb.log(
                    {
                        "SV": self.SV.item(),
                        "Validation loss": self.compute_validation_l2_loss(
                            batch_size
                        ).item(),
                    },
                )

            if loss < eps:
                break

        validation_loss = self.compute_validation_l2_loss(batch_size)
        wandb.log({"Validation loss": validation_loss.item()})
        return validation_loss

    def solve_accumulate(
        self,
        max_iters: int,
        eps: float,
        batch_size: int,
        lambda_reg: float,
        lambda_mes: float,
        lambda_sv: float,
        lambda_bif: float,
        log_every: int,
    ):
        """
        Solving inverse proble one joint at a time and accumulating
        gradient
        """
        iter = 0
        for i in range(max_iters):

            for joint in self.joints:
                self.optimizer.zero_grad()
                loss = 0
                p, d1, d2, merging = joint
                batch_idx = [int(p), int(d1), int(d2)]
                self.solve_arteries(batch_size, batch_idx)
                loss += lambda_reg * self.get_reg_loss(batch_size, batch_idx)
                loss += lambda_mes * self.compute_mesurement_loss(batch_idx)
                loss += lambda_sv * self.compute_SV_loss()
                loss_mass, loss_pressure = self.compute_bifurcation_loss(joint)
                loss += lambda_bif * (loss_mass + loss_pressure)
                loss = loss / len(self.joints)

                loss.backward()

                self.optimizer.step()
            self.propagate_SV()
            print(loss.item())

            if self.track and iter % log_every == 0:
                wandb.log(
                    {
                        "loss": loss.item(),
                        # "reg_loss": lambda_reg * self.get_reg_loss(batch_size).item(),
                        # "mes_loss": lambda_mes * self.compute_mesurement_loss().item(),
                        "SV_loss": lambda_sv * self.compute_SV_loss(),
                        "bif_loss": lambda_bif * (loss_mass + loss_pressure).item(),
                    }
                )
                self.log_arteries()
                wandb.log(
                    {
                        "SV": self.SV.item(),
                        # "Validation loss": self.compute_validation_l2_loss(
                        #     batch_size
                        # ).item(),
                    },
                )

            if loss < eps:
                break

        validation_loss = self.compute_validation_l2_loss(batch_size)
        wandb.log({"Validation loss": validation_loss.item()})

        iter += 1
        return validation_loss
