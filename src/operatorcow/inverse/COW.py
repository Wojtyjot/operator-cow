import gc
import os
import sys
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
from inverse.Find_RCR import find_windkessel
from inverse.ROM.Simulation import estimate_windkessel_func
from inverse.ROM.utils import (
    create_Julia_file,
    create_results_folder,
    create_simulation_script,
    run_simulation,
)
from log_plots import plot_predictions
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
        condition=None,
        cvs: bool = False,
        r0s: torch.Tensor = None,
        p_ref: torch.Tensor = None,
    ):
        # musi tu byc self r0s i propagowane z r0s z cow do arteries w optymalizacji
        self.name = name
        self.g = g
        self.device = device
        self.theta = theta  # NIE JEST ZNORMALIZOWANE !
        self.inputs_f = inputs_f
        self.L = g.ndata["x"][-1, 0]
        self.T = g.ndata["x"][-1, 1]
        # print(f"L = {self.L}")
        # print(f"T = {self.T}")
        self.root = root
        self.mesurement = mesurement
        self.VANO = VANO
        self.condition = condition  # condition for VANO model
        if normalizer_u_bc is not None:
            self.normalizer_u_bc = normalizer_u_bc
        self.a0 = None
        if cvs:
            # TODO
            # tutaj trzeba dodać gdzieś ref pressure.
            # albo set ref pressure w load data
            if r0s is None:
                raise ValueError("r0s must be provided for CVS")
            self.r0s = r0s
            self.p_ref = p_ref
        self.initialize_parameters(mesurement, cvs)

        # del self.g.ndata['y']

    def initialize_parameters(self, measurement: bool, cvs: bool = False):
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
                    self.g.ndata["y"][100 * 25 : 100 * 26, 2].squeeze().to(self.device)
                )

                self.parameters = [self.u_bc_latent]
                self.u_bc_true = self.g.ndata["y"][:100, 2].squeeze()
                if cvs:
                    # jeszcze trzeba zrobic a0 z r0
                    self.theta[29:47] = self.r0s
                    self.a0 = self.compute_a0(self.r0s)
                else:
                    self.a0 = self.g.ndata["y"][0::100, 1].to(self.device)
                self.set_true_u_in(self.g.ndata["y"][0:100, 2].squeeze())
                self.set_true_a_in(self.g.ndata["y"][0:100, 1].squeeze())
                self.set_true_p_in(self.g.ndata["y"][0:100, 0].squeeze())

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
                if cvs:
                    # jeszcze trzeba zrobic a0 z r0
                    self.theta[29:47] = self.r0s
                    self.a0 = self.compute_a0(self.r0s)
                else:
                    self.a0 = self.g.ndata["y"][0::100, 1].to(self.device)
                # self.a0 = self.g.ndata["y"][0::100, 1].to(self.device)
                self.set_true_u_in(self.g.ndata["y"][0:100, 2].squeeze())
                self.set_true_a_in(self.g.ndata["y"][0:100, 1].squeeze())
                self.set_true_p_in(self.g.ndata["y"][0:100, 0].squeeze())

            else:
                self.u_bc_latent = (
                    torch.randn(1, 16).to(self.device).requires_grad_(True)
                )
                # u0 = self.u_bc[0].detach().repeat(200, 1).requires_grad_(True)
                self.parameters = [self.u_bc_latent]
                self.u_bc_true = self.g.ndata["y"][:100, 2].squeeze()
                if cvs:
                    # jeszcze trzeba zrobic a0 z r0
                    self.theta[29:47] = self.r0s
                    self.a0 = self.compute_a0(self.r0s)
                else:
                    self.a0 = self.g.ndata["y"][0::100, 1].to(self.device)
                # self.a0 = self.g.ndata["y"][0::100, 1].to(self.device)
                self.set_true_u_in(self.g.ndata["y"][0:100, 2].squeeze())
                self.set_true_a_in(self.g.ndata["y"][0:100, 1].squeeze())
                self.set_true_p_in(self.g.ndata["y"][0:100, 0].squeeze())

        else:

            if measurement:
                self.u_bc = (
                    self.g.ndata["y"][100 * 25 : 100 * 26, 2]
                    .unsqueeze(-1)
                    .to(self.device)
                    .requires_grad_(True)
                )
                print(self.u_bc.shape)
                # u0 = self.u_bc[0].detach().repeat(200, 1).requires_grad_(True)
                self.mesurement_value = (
                    self.g.ndata["y"][100 * 25 : 100 * 26, 2].squeeze().to(self.device)
                )
                self.parameters = [self.u_bc]
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

    def compute_a0(self, r0s: torch.Tensor):
        """
        Function computes a0 from r0s given indexing of arteries
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
        index = arteries.index(self.name)
        return (torch.pi * r0s[index] ** 2).repeat(50).reshape(-1, 1).to(self.device)

    def get_u_BC(
        self, VANO_model: nn.Module = None, normalizer_u_bc=None, T: float = 1.0
    ):
        """
        Function return boundary condition
        """
        t = (
            torch.linspace(0, self.T, 100).to(self.device).reshape(-1, 1)  # ZMIANA
        )  ## Need to pass T
        x = torch.zeros(100, 1).to(self.device).reshape(-1, 1)
        if self.VANO:
            if self.root:
                self.u_bc_rec = self.u_bc
                return torch.cat((x, t, self.u_bc.reshape(-1, 1)), dim=-1)
            else:
                # need to doceode u_bc and then renormalize
                u_bc = VANO_model.decode(
                    self.parameters[0].reshape(1, 16),
                    self.condition.reshape(1, 11).to(self.device),
                )
                u_bc = normalizer_u_bc.transform(u_bc, inverse=True)
                self.u_bc_rec = u_bc.reshape(-1, 1)
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
        x = torch.linspace(0, self.L, 50).to(self.device).reshape(-1, 1)
        t = torch.zeros(50, 1).to(self.device).reshape(-1, 1)

        if self.root:
            return torch.cat((x, t, self.parameters[0]), dim=-1)
        else:
            return torch.cat((x, t, self.parameters[1]), dim=-1)

    def get_inputs(
        self,
        model: str,
        VANO_model: nn.Module = None,
        normalizer_u_bc=None,
        T: float = 1.0,
    ):
        """
        Function return inputs to model
        """
        if model == "GNOT":
            p0, u0, a0, A_BC, p_BC, p_out_BC = (
                self.inputs_f[1],
                self.inputs_f[2],
                self.inputs_f[3],
                self.inputs_f[4],
                self.inputs_f[5],
                self.inputs_f[6],
            )
            # u0 = self.get_u0()
            in_bc = self.get_u_BC(
                VANO_model=VANO_model, normalizer_u_bc=normalizer_u_bc, T=T
            )
            in_f = MultipleTensors([i for i in (in_bc, a0)])

            return (
                self.g,
                torch.cat((self.theta.to(self.device), self.RT, self.CT)),
                in_f,
            )

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

        plt.plot(self.u_bc_rec.detach().cpu().numpy(), label="reconstructed")
        plt.plot(self.u_bc_true.detach().cpu().numpy(), label="true")
        plt.title(f"U_bc for artery {self.name}")
        plt.legend()
        wandb.log({self.name: plt})
        plt.close()

    def log_mesurement(self):
        """
        Function logs predicted and true value at mesurement point
        """
        plt.plot(self.u_mesurement.detach().cpu().numpy(), label="reconstructed")
        plt.plot(self.mesurement_value.detach().cpu().numpy(), label="true")
        plt.title(f"Mesurement point for artery {self.name}")
        plt.legend()

        wandb.log({f"comparison_{self.name}": plt})

        # plt.show()
        plt.close()

    def set_ROM_mesurement(self):
        """
        Function sets mesurement point for ROM model
        """
        path = (
            Path(os.getcwd())
            .joinpath("Inverse_ROM_results")
            .joinpath(f"{self.name}_u.last")
        )
        u = np.loadtxt(path)
        u = u[:, 1:] * 100  # transform to cm/s
        self.mesurement_value = torch.from_numpy(u[:, 25]).float().to(self.device)
        self.mesurement = True

    def get_u_in_rom(self):
        """
        Function returns u_in for ROM model
        """
        path = (
            Path(os.getcwd())
            .joinpath("Inverse_ROM_results")
            .joinpath(f"{self.name}_u.last")
        )
        u = np.loadtxt(path)
        u = u[:, 1:] * 100
        u = u[:, 0]
        return torch.from_numpy(u).float().to(self.device)

    def get_a_in_rom(self):
        """
        Function returns a_in for ROM model
        """
        path = (
            Path(os.getcwd())
            .joinpath("Inverse_ROM_results")
            .joinpath(f"{self.name}_A.last")
        )
        a = np.loadtxt(path)
        a = a[:, 1:]
        a = a[:, 0] * 1e4
        return torch.from_numpy(a).float().to(self.device)

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
        ### need to print warning that SV should not be used

        self.SV = SV

    def set_RT(self, RT: torch.Tensor):
        self.RT = RT

    def set_CT(self, CT: torch.Tensor):
        self.CT = CT

    def has_mesurement(self):
        return self.mesurement

    def set_reconstructed_u_mesurement(self, u):  # trzeba zmienić nazwy zmiennych
        self.u_mesurement = u

    def get_reconstructed_u_mesurement(self):
        return self.u_mesurement

    def get_true_mesurement(self):
        return self.mesurement_value

    def get_a0(self):
        return self.a0.squeeze()

    def set_a0_rec(self, a0_rec):
        self.a0_rec = a0_rec

    def get_a0_rec(self):
        return self.a0_rec

    def get_true_u_in(self):
        return self.u_in_true

    def set_true_u_in(self, u_in_true):
        self.u_in_true = u_in_true

    def get_true_a_in(self):
        return self.a_in_true

    def set_true_a_in(self, a_in_true):
        self.a_in_true = a_in_true

    def get_true_p_in(self):
        return self.p_in_true

    def set_true_p_in(self, p_in_true):
        self.p_in_true = p_in_true

    def is_root(self):
        return self.root

    def get_T(self):
        return self.T

    def get_r0(self):
        return torch.sqrt(self.a0[0] / torch.pi).detach().cpu().numpy()

    def get_t(self):
        return self.g.ndata["x"][:100, 1]

    def get_L(self):
        return self.L.detach().cpu().numpy()

    def is_outlet(self):
        return self.name in [
            "L_PCA_P2",
            "R_PCA_P2",
            "L_ACA_A2",
            "R_ACA_A2",
            "L_MCA",
            "R_MCA",
        ]

    def set_p_ref(self, p_ref: torch.Tensor):
        """
        Function sets reference pressure computed in initial inverse problem
        pre cvs case
        """
        self.p_ref = p_ref

    def get_p_ref(self):
        return self.p_ref

    def update_theta(self, r0s: torch.Tensor):
        self.theta[29:47] = r0s

    def update_r0s(self, r0s: torch.Tensor):
        self.r0s = r0s
        self.update_theta(r0s)
        self.a0 = self.compute_a0(r0s)

    def set_p_pred(self, pred: torch.Tensor):
        self.p_pred = pred

    def get_p_pred(self):
        return self.p_pred

    def get_u_bc_rec(self):
        return self.u_bc_rec


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
        cvs: bool = False,
        r0s_path: str = None,
        p_ref_path: str = None,
    ):
        self.rho = 1.06  ## must be in CGS units
        # self.SV_true = None
        # self.SV = (
        #    torch.Tensor([np.random.uniform(70, 140)]).to(device).requires_grad_(True)
        # )
        self.RT_true = None
        self.CT_true = None
        self.HR = None
        self.lr = lr
        torch.manual_seed(2137)
        self.RT = (
            torch.Tensor([np.random.uniform(62770839.96, 194904651.43)])
            .to(device)
            .requires_grad_(True)
        )
        self.CT = 1.34 / self.RT.detach()
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
        self.load_data(
            data_path=data_path, cvs=cvs, r0s_path=r0s_path, p_ref_path=p_ref_path
        )
        # self.load_p_ref(p_ref_path=p_ref_path)
        self.optimizer_mes = None
        self.optimizer_non_mes = None
        self.optimizer_full = None
        self.create_optimizer(lr, "RT")
        # self.propagate_SV()
        # print(f"RT: {self.RT_true}")
        # print(f"CT: {self.CT_true}")
        # sys.exit()
        self.propagate_CT()
        self.propagate_RT()
        self.joints = self.create_joints(joints_path)
        # self.r0s = None # Trzbea doać to gdzieś w ładowaniu

    def load_data(
        self,
        data_path: str,
        cvs: bool = False,
        r0s_path: str = None,
        p_ref_path: str = None,
    ):
        """
        Function to load data

        Creates artery objecs ...

        Will read pre specified npy files for testing

        and transform them into usable format

        Parameters
        ----------
        data_path
        : str
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

        if cvs:
            # TRZEBA r0s dodac
            # LOAD r0s
            r0s = np.load(r0s_path, allow_pickle=True)
            r0s = torch.from_numpy(r0s).float()
            self.r0s = r0s
            for artery in arteries:
                X, Y, theta, in_funcs = np.load(
                    data_path + artery + ".npy", allow_pickle=True
                )

                p_ref = np.load(
                    p_ref_path + artery + "_pressure" + ".npy", allow_pickle=True
                )
                p_ref = torch.from_numpy(p_ref).float().to(self.device)
                # if self.SV_true is None:
                #    self.SV_true = theta[-1]
                if self.RT_true is None:
                    self.RT_true = theta[-2]
                if self.CT_true is None:
                    self.CT_true = theta[-1]
                if self.HR is None:
                    self.HR = theta[-3]
                g = dgl.DGLGraph()
                g.add_nodes(X.shape[0])
                L = X[-1, 0]
                T = X[-1, 1]
                g.ndata["x"] = torch.from_numpy(X).float()
                g.ndata["y"] = torch.from_numpy(
                    Y
                ).float()  # y można wyjebać w większości
                # y potrzebny tlko do "pomiarów" bedzie lepiej pamięciowo
                theta = torch.from_numpy(theta).float()
                condition = self.normalizer_theta.transform(
                    theta.to(self.device), inverse=False
                )
                condition = condition[:, :11]
                # passing also true values for comparison
                input_f = [torch.from_numpy(in_func).float() for in_func in in_funcs]
                # moze na poczatek zrobic hardcoded mesurements?
                if artery in ["L_int_carotid_I", "R_int_carotid_I", "Basilar"]:
                    self.arteries.append(
                        Artery(
                            g,
                            input_f,
                            theta[:-2],
                            name=artery,
                            device=self.device,
                            root=True,
                            mesurement=False,
                            VANO=self.VANO,
                            condition=condition,
                            cvs=cvs,
                            r0s=r0s,
                            p_ref=p_ref,
                        )
                    )
                elif artery in [
                    "L_MCA",
                    "R_MCA",
                    "L_ACA_A1",
                    "R_ACA_A1",
                    "L_PCA_P1",
                    "R_PCA_P1",
                    # "L_ACA_A2",
                    # "R_ACA_A2",
                    # "L_PCA_P2",
                    # "R_PCA_P2",
                ]:
                    self.arteries.append(
                        Artery(
                            g,
                            input_f,
                            theta[:-2],
                            name=artery,
                            device=self.device,
                            mesurement=True,
                            VANO=self.VANO,
                            condition=condition,
                            cvs=cvs,
                            r0s=r0s,
                            p_ref=p_ref,
                        )
                    )

                else:
                    self.arteries.append(
                        Artery(
                            g,
                            input_f,
                            theta[:-2],
                            name=artery,
                            device=self.device,
                            mesurement=False,
                            VANO=self.VANO,
                            condition=condition,
                            cvs=cvs,
                            r0s=r0s,
                            p_ref=p_ref,
                        )
                    )  # Theta in arteries is without SV

        else:

            for artery in arteries:
                X, Y, theta, in_funcs = np.load(
                    data_path + artery + ".npy", allow_pickle=True
                )
                # if self.SV_true is None:
                #    self.SV_true = theta[-1]
                if self.RT_true is None:
                    self.RT_true = theta[-2]
                if self.CT_true is None:
                    self.CT_true = theta[-1]
                if self.HR is None:
                    self.HR = theta[-3]
                g = dgl.DGLGraph()
                g.add_nodes(X.shape[0])
                L = X[-1, 0]
                T = X[-1, 1]
                # X[:, 0] = X[:, 0]/L
                # X[:, 1] = X[:, 1]/T
                # in_funcs[0][:,1] = in_funcs[0][:,1]/T
                # in_funcs[3][:,0] = in_funcs[3][:,0]/L
                g.ndata["x"] = torch.from_numpy(X).float()
                g.ndata["y"] = torch.from_numpy(
                    Y
                ).float()  # y można wyjebać w większości
                # y potrzebny tlko do "pomiarów" bedzie lepiej pamięciowo
                # trzeba pamietac by usunac SV i dodawac z cow
                theta = torch.from_numpy(theta).float()
                condition = self.normalizer_theta.transform(
                    theta.to(self.device), inverse=False
                )
                condition = condition[:, :11]
                # passing also true values for comparison
                input_f = [torch.from_numpy(in_func).float() for in_func in in_funcs]
                # moze na poczatek zrobic hardcoded mesurements?
                if artery in ["L_int_carotid_I", "R_int_carotid_I", "Basilar"]:
                    self.arteries.append(
                        Artery(
                            g,
                            input_f,
                            theta[:-2],
                            name=artery,
                            device=self.device,
                            root=True,
                            mesurement=False,
                            VANO=self.VANO,
                            condition=condition,
                        )
                    )
                elif artery in [
                    "L_MCA",
                    "R_MCA",
                    "L_ACA_A1",
                    "R_ACA_A1",
                    "L_PCA_P1",
                    "R_PCA_P1",
                    # "L_ACA_A2",
                    # "R_ACA_A2",
                    # "L_PCA_P2",
                    # "R_PCA_P2",
                ]:
                    self.arteries.append(
                        Artery(
                            g,
                            input_f,
                            theta[:-2],
                            name=artery,
                            device=self.device,
                            mesurement=True,
                            VANO=self.VANO,
                            condition=condition,
                        )
                    )

                else:
                    self.arteries.append(
                        Artery(
                            g,
                            input_f,
                            theta[:-2],
                            name=artery,
                            device=self.device,
                            mesurement=False,
                            VANO=self.VANO,
                            condition=condition,
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

    def load_ref_pressure(self, path: str):
        """
        Function loads reference pressure computed during initial inverse

        ever artery has its own npy file with pressure file

        path: str
            path to folder with npy files
        """
        for artery in self.arteries:
            artery.p_ref = (
                torch.from_numpy(
                    np.load(path + artery.name + ".npy", allow_pickle=True)
                )
                .float()
                .to(self.device)
            )

    def create_optimizer(self, lr: float = 0.5, optim: str = "RT", cvs: bool = False):
        """
        Function creates a single optimizer for all arterial parameters
        """
        # TODO test z LBFGS tylko colsure trzeba zdefiniowac
        if optim not in ["RT", "MES", "NON_MES", "FULL"]:
            raise ValueError("Optimizer not recognized")

        if cvs:
            # trzeba dodać r0 do parametrów i propagować te informacje w theta
            if optim == "RT":
                p = [self.RT.requires_grad_(True)]
                self.optimizer_RT = torch.optim.Adam(p, lr=lr)

            elif optim == "MES":
                p = []

                for artery in self.arteries:
                    if artery.get_parameters() is not None and artery.has_mesurement():
                        p.extend(artery.get_parameters())
                p.extend([self.RT.requires_grad_(True)])
                p.extend(
                    [self.r0s.requires_grad_(True)]
                )  # muszą być dodatnie może torch.exp(r0s)?
                self.optimizer_mes = torch.optim.Adam(p, lr=lr)

            elif optim == "NON_MES":
                p = []
                for artery in self.arteries:
                    if (
                        artery.get_parameters() is not None
                        and not artery.has_mesurement()
                    ):
                        p.extend(artery.get_parameters())
                p.extend([self.RT.requires_grad_(True)])
                p.extend([self.r0s.requires_grad_(True)])
                self.optimizer_non_mes = torch.optim.Adam(p, lr=lr)

            elif optim == "FULL":
                p = []
                for artery in self.arteries:
                    if artery.get_parameters() is not None:
                        p.extend(artery.get_parameters())
                p.extend([self.RT.requires_grad_(True)])
                p.extend([self.r0s.requires_grad_(True)])
                self.optimizer_full = torch.optim.Adam(p, lr=lr)

        else:

            if optim == "RT":
                p = [self.RT.requires_grad_(True)]
                self.optimizer_RT = torch.optim.Adam(p, lr=lr)

            elif optim == "MES":
                p = []

                for artery in self.arteries:
                    if artery.get_parameters() is not None and artery.has_mesurement():
                        p.extend(artery.get_parameters())
                p.extend([self.RT.requires_grad_(True)])
                self.optimizer_mes = torch.optim.Adam(p, lr=lr)

            elif optim == "NON_MES":
                p = []
                for artery in self.arteries:
                    if (
                        artery.get_parameters() is not None
                        and not artery.has_mesurement()
                    ):
                        p.extend(artery.get_parameters())
                p.extend([self.RT.requires_grad_(True)])
                self.optimizer_non_mes = torch.optim.Adam(p, lr=lr)

            elif optim == "FULL":
                p = []
                for artery in self.arteries:
                    if artery.get_parameters() is not None:
                        p.extend(artery.get_parameters())
                p.extend([self.RT.requires_grad_(True)])
                self.optimizer_full = torch.optim.Adam(p, lr=lr)

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
                return out, idx

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
                return out, idx

    def compute_validation_l2_loss(self, batch_size: int):
        """
        Function computes validation loss for all arteries
        """
        loss = 0
        i = 0
        with torch.no_grad():
            for batch, idx in self.loader_GNOT(batch_size):
                i += 1
                g, u_p, g_u = batch
                g, u_p, g_u = (
                    g.to(self.device),
                    u_p.to(self.device),
                    g_u.to(self.device),
                )

                g.ndata["x"] = self.normalizer_x.transform(g.ndata["x"], inverse=False)
                u_p = self.normalizer_theta.transform(u_p, inverse=False)

                g, u_p, g_u = (
                    g.to(self.device),
                    u_p.to(self.device),
                    g_u.to(self.device),
                )

                out = self.model_surrogate(g, u_p, g_u)
                out = self.normalizer_y.transform(out, inverse=True)
                y_true = g.ndata["y"].squeeze().to(self.device)
                los_, _, _ = self.l2_loss(g, out, y_true)
                loss += los_.item()
                out = out.reshape(len(idx), -1, 3)
                y_true = y_true.reshape(len(idx), -1, 3)
                # for index, j in enumerate(idx):
                #    plot_predictions(
                #        out[index, :, :].detach().cpu().numpy(),
                #        y_true[index, :, :].detach().cpu().numpy(),
                #        str(j),
                #    )
        return loss / i

    def compute_bifurcation_loss(self, j=None):
        """
        Function computes bifurcation loss

        conservation of mass and total pressure cont
        """
        # TODO dodać weight to joints
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
                # p1 = self.compute_beta(r0=torch.sqrt(p.get_a_in() / torch.pi)) * (
                #    torch.sqrt(p.get_a_in()) - torch.sqrt(p.get_a_in()[0])
                # )
                # pd1 = self.compute_beta(r0=torch.sqrt(d1.get_a_out() / torch.pi)) * (
                #    torch.sqrt(d1.get_a_out()) - torch.sqrt(d1.get_a_out()[0])
                # )
                # pd2 = self.compute_beta(r0=torch.sqrt(d2.get_a_out() / torch.pi)) * (
                #    torch.sqrt(d2.get_a_out()) - torch.sqrt(d2.get_a_out()[0])
                # )

                # loss_pressure += torch.mean(
                #    torch.square(
                #        p.get_p_in()
                #        + 0.5 * self.rho * torch.square(p.get_u_in())
                #        - d1.get_p_out()
                #        - 0.5 * self.rho * torch.square(d1.get_u_out())
                #    )
                # )
                # loss_pressure += torch.mean(
                #    torch.square(
                #        p.get_p_in()
                #        + 0.5 * self.rho * torch.square(p.get_u_in())
                #        - d2.get_p_out()
                #        - 0.5 * self.rho * torch.square(d2.get_u_out())
                #    )
                # )
                p1 = p.get_p_in()
                pd1 = d1.get_p_out()
                pd2 = d2.get_p_out()
                loss_pressure += torch.mean(torch.square(p1 - pd1)) + torch.mean(
                    torch.square(p1 - pd2)
                )

                # loss_pressure += torch.mean(torch.square(p1 - (pd1 + pd2)))
            else:
                loss_mass += torch.mean(
                    torch.square(
                        p.get_u_out() * p.get_a_out()
                        - d1.get_u_in() * d1.get_a_in()
                        - d2.get_u_in() * d2.get_a_in()
                    )
                )
                # p1 = self.compute_beta(r0 = torch.sqrt(p.get_a_out()/torch.pi))*(torch.sqrt(p.get_a_out()) - torch.sqrt(p.get_a_out()[0]))
                # pd1 = self.compute_beta(r0 = torch.sqrt(d1.get_a_in()/torch.pi))*(torch.sqrt(d1.get_a_in()) - torch.sqrt(d1.get_a_in()[0]))
                # pd2 = self.compute_beta(r0 = torch.sqrt(d2.get_a_in()/torch.pi))*(torch.sqrt(d2.get_a_in()) - torch.sqrt(d2.get_a_in()[0]))

                # loss_pressure += torch.mean(
                #    torch.square(
                #        p.get_p_out()
                #        + 0.5 * self.rho * torch.square(p.get_u_out())
                #        - d1.get_p_in()
                #        - 0.5 * self.rho * torch.square(d1.get_u_in())
                #    )
                # )
                # loss_pressure += torch.mean(
                #    torch.square(
                #        p.get_p_out()
                #        + 0.5 * self.rho * torch.square(p.get_u_out())
                #        - d2.get_p_in()
                #        - 0.5 * self.rho * torch.square(d2.get_u_in())
                #    )
                # )
                p1 = p.get_p_out()
                pd1 = d1.get_p_in()
                pd2 = d2.get_p_in()
                loss_pressure += torch.mean(torch.square(p1 - pd1)) + torch.mean(
                    torch.square(p1 - pd2)
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
                    # p1 = self.compute_beta(r0 = torch.sqrt(p.get_a_in()/torch.pi))*(torch.sqrt(p.get_a_in()) - torch.sqrt(p.get_a_in()[0,:]))
                    # pd1 = self.compute_beta(r0 = torch.sqrt(d1.get_a_out()/torch.pi))*(torch.sqrt(d1.get_a_out()) - torch.sqrt(d1.get_a_out()[0,:]))
                    # pd2 = self.compute_beta(r0 = torch.sqrt(d2.get_a_out()/torch.pi))*(torch.sqrt(d2.get_a_out()) - torch.sqrt(d2.get_a_out()[0,:]))

                    # loss_pressure += torch.mean(
                    #    torch.square(
                    #        p.get_p_in()
                    #        + 0.5 * self.rho * torch.square(p.get_u_in())
                    #        - d1.get_p_out()
                    #        - 0.5 * self.rho * torch.square(d1.get_u_out())
                    #    )
                    # )
                    # loss_pressure += torch.mean(
                    #    torch.square(
                    #        p.get_p_in()
                    #        + 0.5 * self.rho * torch.square(p.get_u_in())
                    #        - d2.get_p_out()
                    #        - 0.5 * self.rho * torch.square(d2.get_u_out())
                    #    )
                    # )
                    p1 = p.get_p_in()
                    pd1 = d1.get_p_out()
                    pd2 = d2.get_p_out()
                    loss_pressure += torch.mean(torch.square(p1 - pd1)) + torch.mean(
                        torch.square(p1 - pd2)
                    )
                else:
                    loss_mass += torch.mean(
                        torch.square(
                            p.get_u_out() * p.get_a_out()
                            - d1.get_u_in() * d1.get_a_in()
                            - d2.get_u_in() * d2.get_a_in()
                        )
                    )
                    # p1 = self.compute_beta(r0 = torch.sqrt(p.get_a_out()/torch.pi))*(torch.sqrt(p.get_a_out()) - torch.sqrt(p.get_a_out()[0,:]))
                    # pd1 = self.compute_beta(r0 = torch.sqrt(d1.get_a_in()/torch.pi))*(torch.sqrt(d1.get_a_in()) - torch.sqrt(d1.get_a_in()[0,:]))
                    # pd2 = self.compute_beta(r0 = torch.sqrt(d2.get_a_in()/torch.pi))*(torch.sqrt(d2.get_a_in()) - torch.sqrt(d2.get_a_in()[0,:]))

                    # loss_pressure += torch.mean(
                    #    torch.square(
                    #        p.get_p_out()
                    #        + 0.5 * self.rho * torch.square(p.get_u_out())
                    #        - d1.get_p_in()
                    #        - 0.5 * self.rho * torch.square(d1.get_u_in())
                    #    )
                    # )
                    # loss_pressure += torch.mean(
                    #    torch.square(
                    #        p.get_p_out()
                    #        + 0.5 * self.rho * torch.square(p.get_u_out())
                    #        - d2.get_p_in()
                    #        - 0.5 * self.rho * torch.square(d2.get_u_in())
                    #    )
                    # )
                    p1 = p.get_p_out()
                    pd1 = d1.get_p_in()
                    pd2 = d2.get_p_in()

                    loss_pressure += torch.mean(torch.square(p1 - pd2)) + torch.mean(
                        torch.square(p1 - pd1)
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

    def compute_pressure_loss(self, batch_idx=None):
        """
        Function computes pressure loss
        """
        # p_ref p predicted duing initial inverse problem
        loss = 0
        if batch_idx is not None:
            for idx in batch_idx:
                loss += nn.MSELoss()(
                    self.arteries[idx].get_p_ref(), self.arteries[idx].get_p_pred()
                )
        else:
            for artery in self.arteries:
                # print(artery.get_p_ref().shape)
                # print(artery.get_p_pred().shape)
                loss += nn.MSELoss()(artery.get_p_ref(), artery.get_p_pred())
        return loss

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
            self.arteries[idx].set_a0_rec(pred[i, 0::100, 1])
            self.arteries[idx].set_p_pred(pred[i, :, 0])
            if self.arteries[idx].has_mesurement():
                self.arteries[idx].set_reconstructed_u_mesurement(
                    pred[i, 100 * 25 : 100 * 26, 2]
                )

    def save_ref_pressure(self, path: str, pred: torch.Tensor, idx: List[int]):
        """
        Function saves ref pressue for cvs prediction
        """
        #### FOR FULLL ONLY NOW
        if len(idx) != len(self.arteries):
            raise ValueError("ONLY FULL COW IDX IMPLEMENTED")

        for i, idx in enumerate(idx):
            np.save(
                path + self.arteries[idx].name + "_pressure" + ".npy",
                pred[i, :, 0].detach().cpu().numpy(),
            )

    def dump_r0s(self, path: str):
        """
        Function saves r0 values from theta as initial guess for cvs inverse
        """
        r0s = []
        for artery in self.arteries:
            r0s.append(artery.get_r0())
        np.save(path + "r0s.npy", r0s)

    def get_arteries(self, idx, model: str = "GNOT"):
        try:
            return self.arteries[idx].get_inputs(
                model, self.model_VANO, self.normalizer_u_bc
            )
        except ValueError:
            pass

    def propagate_RT(self):
        """
        Function passes RT value to arteries
        """

        for artery in self.arteries:
            artery.set_RT(self.RT)

    def update_CT(self):
        """
        Function updates CT value
        """
        self.CT = 1.34 / self.RT.detach()

    def propagate_CT(self):
        """
        Function passes CT value to arteries
        """

        for artery in self.arteries:
            artery.set_CT(self.CT)

    def propagate_r0s(self):
        """
        Function passes r0s to arteries during optimization
        """
        for artery in self.arteries:
            artery.update_r0s(self.r0s)

    def log_arteries(self):
        """
        Function logs all arteries to wandb
        """
        for artery in self.arteries:
            artery.log()

    def compute_a0_loss(self, batch_idx=None):
        loss = 0
        if batch_idx is not None:
            for idx in batch_idx:
                loss += nn.MSELoss()(
                    self.arteries[idx].get_a0_rec(), self.arteries[idx].get_a0()
                )
            return loss
        else:
            for artery in self.arteries:
                # print("REC")
                # print(artery.get_a0_rec().shape)
                # print("TRUE")
                # print(artery.get_a0().shape)
                loss += nn.MSELoss()(artery.get_a0_rec(), artery.get_a0())
            return loss

    def solve_accumulate_2(
        self,
        max_iters: int,
        eps: float,
        batch_size: int,
        lambda_mes: float,
        lambda_mass: float,
        lambda_pressure: float,
        lambda_a0: float,
        run_id=0,
    ):
        """
        Solving inverse proble one joint at a time and accumulating
        gradient
        """
        it = 0
        for i in range(max_iters):
            if self.optimizer_full is not None:
                self.optimizer_full.zero_grad()
            if self.optimizer_non_mes is not None:
                self.optimizer_non_mes.zero_grad()
            if self.optimizer_mes is not None:
                self.optimizer_mes.zero_grad()
            if self.optimizer_RT is not None:
                self.optimizer_RT.zero_grad()

            loss = 0

            _, _ = self.solve_arteries(batch_size)

            if it < 2:
                loss += lambda_mes * self.compute_mesurement_loss()
                loss_mass, loss_pressure = self.compute_bifurcation_loss()
                loss += loss_mass + loss_pressure
                loss_a0 = self.compute_a0_loss()
                loss += loss_a0
                loss.backward()
                self.optimizer_RT.step()
                # print(f"RT = {self.RT}")
                # print(f"CT = {self.CT}")

            elif it < 1000 and it >= 2:
                if self.optimizer_mes is None:
                    self.create_optimizer(self.lr, "MES")

                loss += lambda_mes * self.compute_mesurement_loss()
                loss_mass, loss_pressure = self.compute_bifurcation_loss()
                loss += 1e-20 * (loss_mass + loss_pressure)
                loss_a0 = self.compute_a0_loss()
                loss += lambda_a0 * loss_a0
                loss.backward()
                self.optimizer_mes.step()

            elif it <= 2000 and it >= 1000:
                if self.optimizer_non_mes is None:
                    self.create_optimizer(self.lr, "NON_MES")
                loss_mass, loss_pressure = self.compute_bifurcation_loss()

                # loss += lambda_bif * (1000 * loss_mass + loss_pressure / 1e5)
                loss += lambda_mass * loss_mass
                loss += lambda_pressure * loss_pressure
                loss_a0 = self.compute_a0_loss()
                loss += lambda_a0 * loss_a0
                loss.backward()
                self.optimizer_non_mes.step()

            else:
                if self.optimizer_full is None:
                    self.create_optimizer(self.lr, "FULL")
                loss += lambda_mes * self.compute_mesurement_loss()
                loss_mass, loss_pressure = self.compute_bifurcation_loss()

                # loss += lambda_bif * (1000 * loss_mass +   loss_pressure / 1e5)
                loss += lambda_mass * loss_mass
                loss += lambda_pressure * loss_pressure
                loss_a0 = self.compute_a0_loss()
                loss += lambda_a0 * loss_a0

                try:
                    loss.backward()
                    self.optimizer_full.step()
                except:
                    pass
            if it % 10 == 0:
                pass
                # print(f"Loss = {loss}")

            self.propagate_RT()
            self.update_CT()
            self.propagate_CT()

            it += 1

        validation_loss = self.compute_validation_l2_loss(batch_size)
        # self.optimizer_mes = None
        wandb.log({"Validation loss": validation_loss})
        # self.log_validation()

        # iter += 1
        if False:
            out, idx = self.solve_arteries(batch_size)
            self.save_ref_pressure(
                f"/home/wssk-ptw/Operator/COW_DATASET/CVS_{run_id}/p_ref/", out, idx=idx
            )  # TODO ADD MANUALY
            self.dump_r0s(f"/home/wssk-ptw/Operator/COW_DATASET/CVS_{run_id}/r0s/")
            return validation_loss, loss
        else:
            return validation_loss, loss

    def solve_cvs(
        self,
        max_iters: int,
        eps: float,
        batch_size: int,
        lambda_mes: float,
        lambda_mass: float,
        lambda_pressure: float,
        lambda_a0: float,
        lambda_p_ref: float,
    ):
        """
        Function for solving inverse problem on whole COW with cvs
        opimizing also A0
        """
        it = 0
        for i in range(max_iters):
            if self.optimizer_full is not None:
                self.optimizer_full.zero_grad()
            if self.optimizer_non_mes is not None:
                self.optimizer_non_mes.zero_grad()
            if self.optimizer_mes is not None:
                self.optimizer_mes.zero_grad()
            if self.optimizer_RT is not None:
                self.optimizer_RT.zero_grad()

            loss = 0

            _, _ = self.solve_arteries(batch_size)

            if it < 2:
                loss += lambda_mes * self.compute_mesurement_loss()
                loss_mass, loss_pressure = self.compute_bifurcation_loss()
                loss += loss_mass + loss_pressure
                loss_a0 = self.compute_a0_loss()
                loss += loss_a0
                loss += self.compute_pressure_loss()
                loss.backward()
                self.optimizer_RT.step()
                # print(f"RT = {self.RT}")
                # print(f"CT = {self.CT}")

            elif it < 1000 and it >= 2:
                if self.optimizer_mes is None:
                    self.create_optimizer(self.lr, "MES", cvs=True)

                loss += lambda_mes * self.compute_mesurement_loss()
                loss_mass, loss_pressure = self.compute_bifurcation_loss()
                loss += 1e-20 * (loss_mass + loss_pressure)
                loss_a0 = self.compute_a0_loss()
                # loss += lambda_a0 * loss_a0
                loss_p_ref = self.compute_pressure_loss()
                loss += lambda_p_ref * loss_p_ref
                loss.backward(retain_graph=True)
                self.optimizer_mes.step()
                # print("step")

            elif it <= 2000 and it >= 1000:
                if self.optimizer_non_mes is None:
                    self.create_optimizer(self.lr, "NON_MES", cvs=True)
                loss_mass, loss_pressure = self.compute_bifurcation_loss()

                # loss += lambda_bif * (1000 * loss_mass + loss_pressure / 1e5)
                loss += lambda_mass * loss_mass
                loss += lambda_pressure * loss_pressure
                loss_a0 = self.compute_a0_loss()
                # loss += lambda_a0 * loss_a0
                loss_p_ref = self.compute_pressure_loss()
                loss += lambda_p_ref * loss_p_ref
                loss.backward(retain_graph=True)
                self.optimizer_non_mes.step()

            else:
                if self.optimizer_full is None:
                    self.create_optimizer(self.lr, "FULL", cvs=True)
                loss += lambda_mes * self.compute_mesurement_loss()
                loss_mass, loss_pressure = self.compute_bifurcation_loss()

                # loss += lambda_bif * (1000 * loss_mass +   loss_pressure / 1e5)
                loss += lambda_mass * loss_mass
                loss += lambda_pressure * loss_pressure
                loss_a0 = self.compute_a0_loss()
                # loss += lambda_a0 * loss_a0
                loss_p_ref = self.compute_pressure_loss()
                loss += lambda_p_ref * loss_p_ref

                try:
                    loss.backward(retain_graph=True)
                    self.optimizer_full.step()
                except:
                    pass

            self.propagate_RT()
            self.update_CT()
            self.propagate_CT()
            self.propagate_r0s()

            it += 1

        validation_loss = self.compute_validation_l2_loss(batch_size)
        # self.optimizer_mes = None
        wandb.log({"Validation loss": validation_loss})
        # self.log_validation()

        # iter += 1

        return validation_loss, loss

    def log_validation(self):
        tbl = wandb.Table(
            columns=["Artery", "rL2 Area", "rL2 Pressure", "rL2 Velocity", "rL2 Flow"]
        )
        with torch.no_grad():
            for idx, artery in enumerate(self.arteries):
                for batch, ids in self.loader_GNOT(1, [idx]):
                    g, u_p, g_u = batch
                    g, u_p, g_u = (
                        g.to(self.device),
                        u_p.to(self.device),
                        g_u.to(self.device),
                    )

                    g.ndata["x"] = self.normalizer_x.transform(
                        g.ndata["x"], inverse=False
                    )
                    u_p = self.normalizer_theta.transform(u_p, inverse=False)

                    g, u_p, g_u = (
                        g.to(self.device),
                        u_p.to(self.device),
                        g_u.to(self.device),
                    )

                    out = self.model_surrogate(g, u_p, g_u)
                    out = self.normalizer_y.transform(out, inverse=True)
                    y_true = g.ndata["y"].squeeze().to(self.device)
                    los_a, _, _ = self.l2_loss(g, out[:, 1], y_true[:, 1])
                    los_p, _, _ = self.l2_loss(g, out[:, 0], y_true[:, 0])
                    los_u, _, _ = self.l2_loss(g, out[:, 2], y_true[:, 2])
                    (
                        los_q,
                        _,
                        _,
                    ) = self.l2_loss(
                        g, out[:, 1] * out[:, 2], y_true[:, 1] * y_true[:, 2]
                    )

                    tbl.add_data(
                        artery.name,
                        los_a.item(),
                        los_p.item(),
                        los_u.item(),
                        los_q.item(),
                    )
        wandb.log({"Loss table": tbl})

    def get_validation(self, arteries: dict):
        """
        Function computes validation statistics for plotting
        """
        with torch.no_grad():
            for idx, artery in enumerate(self.arteries):
                for batch, ids in self.loader_GNOT(1, [idx]):
                    g, u_p, g_u = batch
                    g, u_p, g_u = (
                        g.to(self.device),
                        u_p.to(self.device),
                        g_u.to(self.device),
                    )

                    g.ndata["x"] = self.normalizer_x.transform(
                        g.ndata["x"], inverse=False
                    )
                    u_p = self.normalizer_theta.transform(u_p, inverse=False)

                    g, u_p, g_u = (
                        g.to(self.device),
                        u_p.to(self.device),
                        g_u.to(self.device),
                    )

                    out = self.model_surrogate(g, u_p, g_u)
                    out = self.normalizer_y.transform(out, inverse=True)
                    y_true = g.ndata["y"].squeeze().to(self.device)
                    los_a, _, _ = self.l2_loss(g, out[:, 1], y_true[:, 1])
                    los_p, _, _ = self.l2_loss(g, out[:, 0], y_true[:, 0])
                    los_u, _, _ = self.l2_loss(g, out[:, 2], y_true[:, 2])
                    (
                        los_q,
                        _,
                        _,
                    ) = self.l2_loss(
                        g, out[:, 1] * out[:, 2], y_true[:, 1] * y_true[:, 2]
                    )
                    arteries[artery.name]["Area"].append(los_a.item())
                    arteries[artery.name]["Pressure"].append(los_p.item())
                    arteries[artery.name]["Velocity"].append(los_u.item())
                    arteries[artery.name]["Flow"].append(los_q.item())

            return arteries

    def compute_beta(self, r0: float):
        """
        Function computes Eh from empirical relation Olufsen
        """
        k1 = 2e7
        k2 = -22.53
        k3 = 86.5e5
        Eh = r0 * (k1 * torch.exp(k2 * r0) + k3)
        return 4 / 3 * torch.sqrt(torch.Tensor([torch.pi]).to(self.device)) * Eh

    def dump_plots(self, path: str):
        """
        function creates plots of u, a, p  and flow for each artery
        with true and predicted values at inlet with Title of artery name
        and saves them to path with name artery_name.png

        function makes use of artery.get_u_in(), artery.get_a_in(), artery.get_p_in()
        """
        for artery in self.arteries:
            u_in = artery.get_u_in().detach().cpu().numpy()
            a_in = artery.get_a_in().detach().cpu().numpy()
            p_in = artery.get_p_in().detach().cpu().numpy()
            u_in_true = artery.get_true_u_in().detach().cpu().numpy()
            a_in_true = artery.get_true_a_in().detach().cpu().numpy()
            p_in_true = artery.get_true_p_in().detach().cpu().numpy()

            fig, axs = plt.subplots(2, 2, figsize=(10, 10))
            axs[0, 0].plot(u_in, label="Predicted")
            axs[0, 0].plot(u_in_true, label="True")
            axs[0, 0].set_title("Velocity")
            axs[0, 0].set_ylabel("cm/s")
            axs[0, 0].set_xlabel("time_step")
            axs[0, 0].legend()
            axs[0, 0].grid()
            axs[0, 1].plot(a_in, label="Predicted")
            axs[0, 1].plot(a_in_true, label="True")
            axs[0, 1].set_title("Area")
            axs[0, 1].set_ylabel("cm^2")
            axs[0, 1].set_xlabel("time_step")
            y_lim = axs[0, 1].get_ylim()
            axs[0, 1].set_ylim(y_lim[0] - 0.10 * y_lim[0], y_lim[1] + 0.10 * y_lim[1])
            axs[0, 1].legend()
            axs[0, 1].grid()
            axs[1, 0].plot(p_in, label="Predicted")
            axs[1, 0].plot(p_in_true, label="True")
            axs[1, 0].set_title("Pressure")
            axs[1, 0].set_ylabel("Baye")
            axs[1, 0].set_xlabel("time_step")
            axs[1, 0].legend()
            axs[1, 0].grid()
            axs[1, 1].plot(a_in * u_in, label="Predicted")
            axs[1, 1].plot(a_in_true * u_in_true, label="True")
            axs[1, 1].set_title("Flow")
            axs[1, 1].set_ylabel("cm^3/s")
            axs[1, 1].set_xlabel("time_step")
            axs[1, 1].legend()
            axs[1, 1].grid()
            fig.suptitle(artery.name)
            plt.savefig(os.path.join(path, f"{artery.name}.png"))
            plt.close()

    def dump_ROM_plots(self, path: str):
        """
        Funtion plots comparison etween estimated ROM simulations and true values
        for velocity area and flow
        """
        for artery in self.arteries:
            u_in = artery.get_u_in().detach().cpu().numpy()
            a_in = artery.get_a_in().detach().cpu().numpy()
            # p_in = artery.get_p_in().detach().cpu().numpy()
            u_in_true = artery.get_true_u_in().detach().cpu().numpy()
            a_in_true = artery.get_true_a_in().detach().cpu().numpy()
            # p_in_true = artery.get_true_p_in().detach().cpu().numpy()
            u_in_rom = artery.get_u_in_rom().detach().cpu().numpy()
            a_in_rom = artery.get_a_in_rom().detach().cpu().numpy()
            # p_in_rom = artery.get_p_in_rom().detach().cpu().numpy()

            fig, axs = plt.subplots(2, 2, figsize=(10, 10))
            axs[0, 0].plot(u_in, label="Predicted")
            axs[0, 0].plot(u_in_true, label="True")
            axs[0, 0].plot(u_in_rom, label="ROM")
            axs[0, 0].set_title("Velocity")
            axs[0, 0].set_ylabel("cm/s")
            axs[0, 0].set_xlabel("time_step")
            axs[0, 0].legend()
            axs[0, 1].plot(a_in, label="Predicted")
            axs[0, 1].plot(a_in_true, label="True")
            axs[0, 1].plot(a_in_rom, label="ROM")
            axs[0, 1].set_title("Area")
            axs[0, 1].set_ylabel("cm^2")
            axs[0, 1].set_xlabel("time_step")
            axs[0, 1].legend()
            # axs[1, 0].plot(p_in, label="Predicted")
            # axs[1, 0].plot(p_in_true, label="True")
            # axs[1, 0].plot(p_in_rom, label="ROM")
            # axs[1, 0].set_title("Pressure")
            # axs[1, 0].set_ylabel("Baye")
            # axs[1, 0].set_xlabel("time_step")
            axs[1, 0].legend()
            axs[1, 1].plot(a_in * u_in, label="Predicted")
            axs[1, 1].plot(a_in_true * u_in_true, label="True")
            axs[1, 1].plot(a_in_rom * u_in_rom, label="ROM")
            axs[1, 1].set_title("Flow")
            axs[1, 1].set_ylabel("cm^3/s")
            axs[1, 1].set_xlabel("time_step")
            axs[1, 1].legend()
            fig.suptitle(artery.name)
            plt.savefig(os.path.join(path, f"{artery.name}_ROM.png"))
            plt.close()

    def dump_params(self, path: str):
        """
        Function dumps parameters of simulation HR, RT_true, CT_true, RT rec, CT rec
        to txt file
        """
        with open(os.path.join(path, "params.txt"), "w") as f:
            f.write(f"HR = {self.HR}\n")
            f.write(f"RT_true = {self.RT_true}\n")
            f.write(f"CT_true = {self.CT_true}\n")
            f.write(f"RT_rec = {self.RT}\n")
            f.write(f"CT_rec = {self.CT}\n")

    def dump_statistics(self, path: str):
        """
        Function dumps statistics of predicted and true values of u, a, p

        Pi, MFV, flow, u_max, u_min, flow_max, flow_min,
        """
        with open(os.path.join(path, "statistics.txt"), "w") as f:
            for artery in self.arteries:
                u_in = artery.get_u_in().detach().cpu().numpy()
                a_in = artery.get_a_in().detach().cpu().numpy()
                p_in = artery.get_p_in().detach().cpu().numpy()
                u_in_true = artery.get_true_u_in().detach().cpu().numpy()
                a_in_true = artery.get_true_a_in().detach().cpu().numpy()
                p_in_true = artery.get_true_p_in().detach().cpu().numpy()
                u_max = np.max(u_in)
                u_min = np.min(u_in)
                a_max = np.max(a_in)
                a_min = np.min(a_in)
                p_max = np.max(p_in)
                p_min = np.min(p_in)
                u_max_true = np.max(u_in_true)
                u_min_true = np.min(u_in_true)
                MFV_pred = np.min(u_in) + (np.max(u_in) - np.min(u_in)) / 3
                PI_pred = (np.max(u_in) - np.min(u_in)) / MFV_pred
                flow_mean_pred = np.mean(a_in * u_in)
                flow_max_pred = np.max(a_in * u_in)
                flow_min_pred = np.min(a_in * u_in)
                MFV_true = (
                    np.min(u_in_true) + (np.max(u_in_true) - np.min(u_in_true)) / 3
                )
                PI_true = (np.max(u_in_true) - np.min(u_in_true)) / MFV_true
                flow_mean_true = np.mean(a_in_true * u_in_true)
                flow_max_true = np.max(a_in_true * u_in_true)
                flow_min_true = np.min(a_in_true * u_in_true)

                f.write(f"Artery {artery.name}\n")
                f.write(f"Predicted\n")
                f.write(f"u_max = {u_max}\n")
                f.write(f"u_min = {u_min}\n")
                f.write(f"p_max = {p_max}\n")
                f.write(f"p_min = {p_min}\n")
                f.write(f"flow_mean = {flow_mean_pred}\n")
                f.write(f"flow_max = {flow_max_pred}\n")
                f.write(f"flow_min = {flow_min_pred}\n")
                f.write(f"MFV = {MFV_pred}\n")
                f.write(f"PI = {PI_pred}\n")
                f.write(f"True\n")
                f.write(f"u_max = {u_max_true}\n")
                f.write(f"u_min = {u_min_true}\n")
                f.write(f"p_max = {p_max}\n")
                f.write(f"p_min = {p_min}\n")
                f.write(f"flow_mean = {flow_mean_true}\n")
                f.write(f"flow_max = {flow_max_true}\n")
                f.write(f"flow_min = {flow_min_true}\n")
                f.write(f"MFV = {MFV_true}\n")
                f.write(f"PI = {PI_true}\n")
                f.write("differeces\n")
                f.write(f"u_max_diff = {u_max - u_max_true}\n")
                f.write(f"u_min_diff = {u_min - u_min_true}\n")
                f.write(f"p_max_diff = {p_max - p_max}\n")
                f.write(f"p_min_diff = {p_min - p_min}\n")
                f.write(f"flow_mean_diff = {flow_mean_pred - flow_mean_true}\n")
                f.write(f"flow_max_diff = {flow_max_pred - flow_max_true}\n")
                f.write(f"flow_min_diff = {flow_min_pred - flow_min_true}\n")
                f.write(f"MFV_diff = {MFV_pred - MFV_true}\n")
                f.write(f"PI_diff = {PI_pred - PI_true}\n")

    def dump_validation(self, path: str, arteries: dict):
        """
        function dumps validation statistics to path
        """
        with open(os.path.join(path, "validation.txt"), "w") as f:
            for artery in self.arteries:
                f.write(f"Artery {artery.name}\n")
                f.write(f"Area\n")
                f.write(
                    f"Relative L2 error = {np.mean(arteries[artery.name]['Area'])}\n"
                )
                f.write(f"Pressure\n")
                f.write(
                    f"Relative L2 error = {np.mean(arteries[artery.name]['Pressure'])}\n"
                )
                f.write(f"Velocity\n")
                f.write(
                    f"Relative L2 error = {np.mean(arteries[artery.name]['Velocity'])}\n"
                )
                f.write(f"Flow\n")
                f.write(
                    f"Relative L2 error = {np.mean(arteries[artery.name]['Flow'])}\n"
                )

    def dump_reconstructed_u_bc_plots(self, path: str):
        """
        Function dumps reconstructed u values to path
        """
        for artery in self.arteries:
            u_in = artery.get_u_bc_rec().detach().cpu().numpy()
            u_in_true = artery.get_true_u_in().detach().cpu().numpy()

            fig, axs = plt.subplots(1, 1, figsize=(10, 10))
            axs.plot(u_in, label="Rec")
            axs.plot(u_in_true, label="True")
            axs.set_title("Velocity")
            axs.set_ylabel("cm/s")
            axs.set_xlabel("time_step")
            axs.legend()
            axs.grid()
            fig.suptitle(artery.name)
            plt.savefig(os.path.join(path, f"{artery.name}_u_bc.png"))
            plt.close()

    def get_num_arteries(self):
        """
        Function returns number of arteries in cow
        """
        return len(self.arteries)

    def get_artery(self, idx):
        """
        Returns artery with idx
        """
        return self.arteries[idx]

    def get_r0s(self):
        """
        Function returns r0s for all arteries

        and transforms from cm -=> m
        """
        r0s = list()
        for artery in self.arteries:
            r0s.append(artery.get_r0() * 1e-2)
        return r0s

    def get_Ls(self):
        """
        Function return lengths of all arteries
        and transforms them from cm -> m

        Returns:
            list: A list of artery lengths in meters.
        """
        Ls = list()
        for artery in self.arteries:
            Ls.append(artery.get_L() * 1e-2)
        return Ls

    ### need some function for outlet predictions

    def get_outlet_predictions(self, true: bool = False):
        """
        Function returns dict with outlet predictions
        """
        out = dict()
        for idx, artery in enumerate(self.arteries):
            if artery.is_outlet():
                # need to transfrom to si units
                if true:
                    out[idx] = (
                        artery.get_true_u_in().detach().cpu().numpy() * 1e-4,
                        artery.get_true_a_in().detach().cpu().numpy() * 1e-2,
                        artery.get_true_p_in().detach().cpu().numpy() / 10,
                        artery.get_t().detach().cpu().numpy(),
                    )

                else:

                    out[idx] = (
                        artery.get_a_out().detach().cpu().numpy() * 1e-4,
                        artery.get_u_out().detach().cpu().numpy() * 1e-2,
                        artery.get_p_out().detach().cpu().numpy() / 10,
                        artery.get_t().detach().cpu().numpy(),
                    )
        return out

    def create_inlet_file(self, project_name):
        """
        Function creates inlet dat file for openBF simulation

        needs flow in m^3/s andtimesteps in seconds
        """
        # project_name = project_name + "_results"
        path = Path(os.getcwd())  # .joinpath(project_name)
        for artery in self.arteries:
            if artery.is_root():
                u_in = artery.get_u_in().detach().cpu().numpy()
                a_in = artery.get_a_in().detach().cpu().numpy()
                flow = u_in * a_in  # in cm^3/s need to convert
                flow = flow * 1e-6
                T = artery.get_T()
                t = np.linspace(0, T, len(flow))
                # save_folder_path = "/home/wssk-ptw/Operator/operator-cow/src/operatorcow/inverse/ROM/ROM_DATA/"

                with open(path.joinpath(f"{artery.name}_inlet.dat"), "w") as f:
                    for i in range(len(flow)):
                        f.write(f"{t[i]} {flow[i]}\n")

    def ROM_simulation(self, csv_path: str, R2, C, Z):
        """
        Function estimates windessel parameters for arteries,
        performs rom simulation and creates fake "mesurements"
        for AcoA and pcoms
        """
        create_results_folder(project_name="Inverse_ROM")
        self.create_inlet_file(project_name="Inverse_ROM")
        df = pd.read_csv(csv_path)
        #### need to create df that is base for creatig scripts etc
        r0s = self.get_r0s()
        Ls = self.get_Ls()
        df["Rp"] = r0s
        df["Rd"] = r0s
        # df["R1"] = Z
        # df["R2"] = R2
        # df["C"] = C
        df["L"] = Ls
        df = estimate_windkessel_func(df, self.RT.detach().cpu().numpy(), 1050)

        print("Creating simulation script...")
        create_simulation_script(df, project_name="Inverse_ROM")
        print("Simulation script created")
        print("Creating Julia file")
        create_Julia_file(
            project_name="Inverse_ROM",
            src="/home/wssk-ptw/Operator/operator-cow/src/operatorcow/inverse/ROM/src",
        )
        print("Julia file created")
        print("Running simulation")
        run_simulation(project_name="Inverse_ROM")
        print("Simulation finished")

    def set_ROM_mesurement(self):
        """
        Function assigns fake mesurement to AcoA and pcoms
        from ROM simulation
        """
        for artery in self.arteries:
            if artery.name in [
                "AcoA",
                "L_PcoA",
                "R_PcoA",
                "L_ACA_A2",
                "R_ACA_A2",
                "L_PCA_P2",
                "R_PCA_P2",
            ]:
                artery.set_ROM_mesurement()

    def purge_ROM_mesurements(self):
        """
        Function purges fake mesurements
        """
        for artery in self.arteries:
            if artery.name in [
                "AcoA",
                "L_PcoA",
                "R_PcoA",
                "L_ACA_A2",
                "R_ACA_A2",
                "L_PCA_P2",
                "R_PCA_P2",
            ]:
                artery.mesurement = False  # moze jakis setter zrobic


class Multiple_COWs(object):
    """
    Super class for executing multiple instances of COW in parralel

    Changes include computing solve arteries for multiple instances in one pass

    changes include bathcing from all arteries in all instances
    """

    def __init__(
        self, COWs: List[COW], normalizer_x, normalizer_theta, model_surrogate
    ):
        self.COWs = COWs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.normalizer_x = normalizer_x
        self.normalizer_theta = normalizer_theta
        self.model_surrogate = model_surrogate

    def initialize_losses(self):
        self.losses = list(0 for i in range(len(self.COWs)))
        self.validation = list()

    def loader(self, batch_size: int):
        """
        Loader for multiple cows, batch size represents number of cows sampled

        batch_size = len(COWs)???
        """

        batch_idx = [list(range(18)) for i in range(batch_size)]
        transposed = zip(
            *[
                self.COWs[idx_cow].get_arteries(idx_a)
                for idx_cow, idx_a in enumerate(batch_idx)
            ]
        )
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
        yield batched, batch_idx

    def solve_cows(self):
        for batch, idx in self.loader(len(self.COWs)):
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
            out = out.reshape(len(self.COWs), 18, -1, 3)  # mam nadzieje ze to dobrze

            # tu musi byc funkcja do zapisu wynikow do artery
            # print(idx)
            for idx_cow, idx_artery in enumerate(idx):
                self.COWs[idx_cow].update_arteries(out[idx_cow].squeeze(), idx_artery)

            return out, idx

    def solve_inverse(
        self,
        max_iters: int,
        eps: float,
        batch_size: int,
        lambda_mes: float,
        lambda_mass: float,
        lambda_pressure: float,
        lambda_a0: float,
        run_id=0,
    ):
        """
        Optimization loop for multile cows
        """
        it = 0
        for i in range(max_iters):
            _, _ = self.solve_cows()
            self.initialize_losses()
            for idx_cow in len(self.COWs):
                if self.COWs[idx_cow].optimizer_full is not None:
                    self.COWs[idx_cow].optimizer_full.zero_grad()
                if self.COWs[idx_cow].optimizer_non_mes is not None:
                    self.COWs[idx_cow].optimizer_non_mes.zero_grad()
                if self.COWs[idx_cow].optimizer_mes is not None:
                    self.COWs[idx_cow].optimizer_mes.zero_grad()
                if self.COWs[idx_cow].optimizer_RT is not None:
                    self.COWs[idx_cow].optimizer_RT.zero_grad()

                if it < 2:
                    self.losses[idx_cow] += (
                        lambda_mes * self.COWs[idx_cow].compute_mesurement_loss()
                    )
                    loss_mass, loss_pressure = self.COWs[
                        idx_cow
                    ].compute_bifurcation_loss()
                    self.losses[idx_cow] += loss_mass + loss_pressure
                    loss_a0 = self.COWs[idx_cow].compute_a0_loss()
                    self.losses[idx_cow] += loss_a0
                    self.losses[idx_cow].backward()
                    self.COWs[idx_cow].optimizer_RT.step()
                    # print(f"RT = {self.RT}")
                    # print(f"CT = {self.CT}")

                elif it < 1000 and it >= 2:
                    if self.COWs[idx_cow].optimizer_mes is None:
                        self.COWs[idx_cow].create_optimizer(self.lr, "MES")
                    self.losses[idx_cow] += (
                        lambda_mes * self.COWs[idx_cow].compute_mesurement_loss()
                    )
                    loss_mass, loss_pressure = self.COWs[
                        idx_cow
                    ].compute_bifurcation_loss()
                    self.losses[idx_cow] += 1e-20 * (loss_mass + loss_pressure)
                    loss_a0 = self.COWs[idx_cow].compute_a0_loss()
                    self.losses[idx_cow] += lambda_a0 * loss_a0
                    self.losses[idx_cow].backward()
                    self.COWs[idx_cow].optimizer_mes.step()

                elif it <= 2000 and it >= 1000:
                    if self.COWs[idx_cow].optimizer_non_mes is None:
                        self.COWs[idx_cow].create_optimizer(self.lr, "NON_MES")
                    loss_mass, loss_pressure = self.COWs[
                        idx_cow
                    ].compute_bifurcation_loss()

                    # loss += lambda_bif * (1000 * loss_mass + loss_pressure / 1e5)
                    self.losses[idx_cow] += lambda_mass * loss_mass
                    self.losses[idx_cow] += lambda_pressure * loss_pressure
                    loss_a0 = self.COWs[idx_cow].compute_a0_loss()
                    self.losses[idx_cow] += lambda_a0 * loss_a0
                    self.losses[idx_cow].backward()
                    self.COWs[idx_cow].optimizer_non_mes.step()

                else:
                    if self.COWs[idx_cow].optimizer_full is None:
                        self.COWs[idx_cow].create_optimizer(self.lr, "FULL")
                    self.losses[idx_cow] += (
                        lambda_mes * self.COWs[idx_cow].compute_mesurement_loss()
                    )
                    loss_mass, loss_pressure = self.COWs[
                        idx_cow
                    ].compute_bifurcation_loss()

                    # loss += lambda_bif * (1000 * loss_mass +   loss_pressure / 1e5)
                    self.losses[idx_cow] += lambda_mass * loss_mass
                    self.losses[idx_cow] += lambda_pressure * loss_pressure
                    loss_a0 = self.COWs[idx_cow].compute_a0_loss()
                    self.losses[idx_cow] += lambda_a0 * loss_a0

                    try:
                        self.losses[idx_cow].backward()
                        self.COWs[idx_cow].optimizer_full.step()
                    except:
                        pass
                if it % 10 == 0:
                    print(f"Loss = {self.losses[idx_cow]}")

                self.COWs[idx_cow].propagate_RT()
                self.COWs[idx_cow].update_CT()
                self.COWs[idx_cow].propagate_CT()
                it += 1

        for idx_cow in len(self.COWs):
            self.validation.append(self.COWs[idx_cow].compute_validation_l2_loss(18))

        for idx_cow in len(self.COWs):
            print(f"Validation loss for cow {idx_cow} = {self.validation[idx_cow]}")
            print(f"Loss for cow {idx_cow} = {self.losses[idx_cow]}")
