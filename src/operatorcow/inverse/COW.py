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
from log_plots import plot_predictions, unit_to_mmHg
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
    Class Artery
    This class represents an artery in a model, storing data and parameters associated with the artery.
    Attributes:
        name (str): Name of the artery.
        g (dgl.DGLGraph): Graph representation of the artery.
        device (str): Device to run the computations on.
        theta (torch.Tensor): Parameter tensor (not normalized).
        inputs_f (List[torch.Tensor]): List of input tensors.
        L (float): Length of the artery.
        T (float): Time parameter.
        root (bool): Indicates if the artery is a root artery.
        mesurement (bool): Indicates if measurements are available.
        VANO (bool): Indicates if VANO model is used.
        condition (optional): Condition for VANO model.
        normalizer_u_bc (optional): Normalizer for boundary condition.
        a0 (torch.Tensor): Initial condition for a0.
    Methods:
        initialize_parameters(measurement: bool):
            Initializes parameters to optimize.
        get_parameters() -> List[torch.Tensor]:
            Returns parameters to optimize.
        compute_a0(r0s: torch.Tensor) -> torch.Tensor:
            Computes a0 from r0s given indexing of arteries.
        get_u_BC(VANO_model: nn.Module = None, normalizer_u_bc=None, T: float = 1.0) -> torch.Tensor:
            Returns boundary condition.
        get_u0() -> torch.Tensor:
            Returns u0 initial condition.
        input_a0() -> torch.Tensor:
            Returns a0 initial condition.
        get_inputs(model: str, VANO_model: nn.Module = None, normalizer_u_bc=None, T: float = 1.0) -> Tuple:
            Returns inputs to model.
        log():
            Logs artery parameters to wandb.
        log_mesurement():
            Logs predicted and true value at measurement point.
        dump_mesurement(fig_path: str):
            Dumps predicted and true value at measurement point.
        set_u_in(u_in: torch.Tensor):
            Sets u_in.
        get_u_in() -> torch.Tensor:
            Returns u_in.
        set_u_out(u_out: torch.Tensor):
            Sets u_out.
        get_u_out() -> torch.Tensor:
            Returns u_out.
        set_a_in(a_in: torch.Tensor):
            Sets a_in.
        get_a_in() -> torch.Tensor:
            Returns a_in.
        set_a_out(a_out: torch.Tensor):
            Sets a_out.
        get_a_out() -> torch.Tensor:
            Returns a_out.
        set_p_in(p_in: torch.Tensor):
            Sets p_in.
        get_p_in() -> torch.Tensor:
            Returns p_in.
        set_p_out(p_out: torch.Tensor):
            Sets p_out.
        get_p_out() -> torch.Tensor:
            Returns p_out.
        set_RT(RT: torch.Tensor):
            Sets RT.
        set_CT(CT: torch.Tensor):
            Sets CT.
        has_mesurement() -> bool:
            Returns if measurements are available.
        set_reconstructed_u_mesurement(u: torch.Tensor):
            Sets reconstructed u measurement.
        get_reconstructed_u_mesurement() -> torch.Tensor:
            Returns reconstructed u measurement.
        get_true_mesurement() -> torch.Tensor:
            Returns true measurement value.
        get_a0() -> torch.Tensor:
            Returns a0.
        set_a0_rec(a0_rec: torch.Tensor):
            Sets a0_rec.
        get_a0_rec() -> torch.Tensor:
            Returns a0_rec.
        get_true_u_in() -> torch.Tensor:
            Returns true u_in.
        set_true_u_in(u_in_true: torch.Tensor):
            Sets true u_in.
        get_true_a_in() -> torch.Tensor:
            Returns true a_in.
        set_true_a_in(a_in_true: torch.Tensor):
            Sets true a_in.
        get_true_p_in() -> torch.Tensor:
            Returns true p_in.
        set_true_p_in(p_in_true: torch.Tensor):
            Sets true p_in.
        is_root() -> bool:
            Returns if the artery is a root artery.
        get_T() -> float:
            Returns T.
        get_r0() -> float:
            Returns r0.
        get_t() -> torch.Tensor:
            Returns time tensor.
        get_L() -> float:
            Returns L.
        is_outlet() -> bool:
            Returns if the artery is an outlet.
        set_p_pred(pred: torch.Tensor):
            Sets predicted pressure.
        get_p_pred() -> torch.Tensor:
            Returns predicted pressure.
        get_u_bc_rec() -> torch.Tensor:
            Returns reconstructed boundary condition.
        get_latent() -> torch.Tensor:
            Returns latent boundary condition.
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
    ):
        # musi tu byc self r0s i propagowane z r0s z cow do arteries w optymalizacji
        self.name = name
        self.g = g
        self.device = device
        self.theta = theta  # NIE JEST ZNORMALIZOWANE !
        self.inputs_f = inputs_f
        self.L = g.ndata["x"][-1, 0]
        self.T = g.ndata["x"][-1, 1]

        self.root = root
        self.mesurement = mesurement
        self.VANO = VANO
        self.condition = condition  # condition for VANO model
        if normalizer_u_bc is not None:
            self.normalizer_u_bc = normalizer_u_bc
        self.a0 = None
        self.initialize_parameters(mesurement)

    def initialize_parameters(self, measurement: bool):
        """
        Initialize parameters for optimization based on measurement and root conditions.
        Parameters:
        -----------
        measurement : bool
            Indicates whether the measurement is assumed to be in the middle of the artery.
        Attributes:
        -----------
        u_bc_latent : torch.Tensor
            Latent boundary condition tensor, initialized with random values and requires gradient.
        u_bc : torch.Tensor
            Boundary condition tensor, initialized from graph data and requires gradient.
        mesurement_value : torch.Tensor
            Measurement value tensor, extracted from graph data.
        parameters : list
            List of parameters to optimize.
        u_bc_true : torch.Tensor
            True boundary condition tensor, extracted from graph data.
        a0 : torch.Tensor
            Initial value tensor for 'a', extracted from graph data.
        Notes:
        ------
        - If `self.VANO` is True, different initialization strategies are applied based on the `measurement` and `root` flags.
        - If `self.VANO` is False, different initialization strategies are applied based on the `measurement` and `root` flags.
        - Noise addition to measurements is commented out but can be enabled if needed.
        """

        if self.VANO:
            if measurement:

                self.u_bc_latent = (
                    torch.randn(1, 16).to(self.device).requires_grad_(True)
                )

                # ADDED NOISE TO MEASUREMENT
                # noise = torch.normal(0,2.5, self.g.ndata["y"][100 * 25 : 100 * 26, 2].squeeze().shape).to(self.device)
                # self.noise = noise
                self.mesurement_value = (
                    self.g.ndata["y"][100 * 25 : 100 * 26, 2]
                    .squeeze()
                    .to(self.device)
                    # + noise #torch.normal(0,2.5, self.g.ndata["y"][100 * 25 : 100 * 26, 2].squeeze().shape).to(self.device)
                )

                self.parameters = [self.u_bc_latent]
                self.u_bc_true = self.g.ndata["y"][:100, 2].squeeze()  # + noise.cpu()

                self.a0 = self.g.ndata["y"][0::100, 1].to(self.device)
                self.set_true_u_in(self.g.ndata["y"][0:100, 2].squeeze())
                self.set_true_a_in(self.g.ndata["y"][0:100, 1].squeeze())
                self.set_true_p_in(self.g.ndata["y"][0:100, 0].squeeze())

            elif self.root:
                # ADDED NOISE TO MEASUREMENT

                # noise = torch.normal(0,2.5, self.g.ndata["y"][0:100, 2].unsqueeze(-1).shape).to(self.device)
                # self.noise = noise
                self.u_bc = (
                    self.g.ndata["y"][0:100, 2]
                    .unsqueeze(-1)
                    .to(self.device)
                    # .requires_grad_(True)
                    # + noise #torch.normal(0,2.5, self.g.ndata["y"][0:100, 2].unsqueeze(-1).shape).to(self.device)
                ).requires_grad_(True)

                # u0 = self.u_bc[0].detach().repeat(200, 1).requires_grad_(True)
                # self.parameters = [u0]
                self.u_bc_true = self.g.ndata["y"][
                    :100, 2
                ].squeeze()  # + noise.squeeze().cpu()
                self.parameters = None

                self.a0 = self.g.ndata["y"][0::100, 1].to(self.device)

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

                self.a0 = self.g.ndata["y"][0::100, 1].to(self.device)

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
        Retrieve the parameters of the object.
        Returns:
            dict: A dictionary containing the parameters.
        """

        return self.parameters

    def compute_a0(self, r0s: torch.Tensor):
        """
        Compute the cross-sectional area (a0) for a specific artery.
        This function calculates the cross-sectional area (a0) for a specific artery
        based on the radius tensor `r0s`. The artery is identified by the `self.name`
        attribute, which should match one of the predefined artery names in the `arteries` list.
        Args:
            r0s (torch.Tensor): A tensor containing the radii of the arteries.
        Returns:
            torch.Tensor: A tensor containing the computed cross-sectional area (a0)
                          for the specified artery, repeated 50 times and reshaped
                          to a column vector. The tensor is moved to the device specified
                          by `self.device`.
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

    def get_u_BC(self, VANO_model: nn.Module = None, normalizer_u_bc=None):
        """
        Generates boundary condition tensor for the model.
        Args:
            VANO_model (nn.Module, optional): The VANO model used for decoding boundary conditions. Defaults to None.
            normalizer_u_bc (optional): Normalizer for boundary condition values. Defaults to None.
        Returns:
            torch.Tensor: A tensor containing concatenated x, t, and boundary condition values.
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
        Generates the initial condition tensor `u0` for the model.
        This method creates a tensor `u0` by concatenating spatial coordinates `x`,
        temporal coordinates `t`, and a parameter tensor from `self.parameters`
        based on the value of `self.root`.
        Returns:
            torch.Tensor: The concatenated tensor `u0` with shape (50, 3).
        """

        x = torch.linspace(0, self.L, 50).to(self.device).reshape(-1, 1)
        t = torch.zeros(50, 1).to(self.device).reshape(-1, 1)

        if self.root:
            return torch.cat((x, t, self.parameters[0]), dim=-1)
        else:
            return torch.cat((x, t, self.parameters[1]), dim=-1)

    def input_a0(self):
        """
        Generates the initial condition tensor `a0`.
        This function creates a tensor representing the initial condition `a0` by
        combining a linearly spaced tensor `x` and a zero tensor `t` with the
        reshaped `a0` attribute of the class instance.
        Returns:
            torch.Tensor: A tensor of shape (50, 3) containing the initial condition `a0`.
        """

        x = torch.linspace(0, self.L, 50).to(self.device).reshape(-1, 1)

        t = torch.zeros(50, 1).to(self.device).reshape(-1, 1)

        a0 = torch.cat((x, t, self.a0.reshape(-1, 1)), dim=-1)

        return a0

    def get_inputs(
        self,
        model: str,
        VANO_model: nn.Module = None,
        normalizer_u_bc=None,
    ):
        """
        Returns the inputs to the specified model.
        Parameters:
        -----------
        model : str
            The name of the model. Currently, only "GNOT" is supported.
        VANO_model : nn.Module, optional
            The VANO model used for boundary conditions. Default is None.
        normalizer_u_bc : optional
            Normalizer for boundary conditions. Default is None.
        Returns:
        --------
        tuple
            A tuple containing:
            - self.g
            - Concatenated tensor of self.theta, self.RT, and self.CT
            - in_f, which is a MultipleTensors object containing in_bc and a0
        Raises:
        -------
        ValueError
            If the model name is not recognized (i.e., not "GNOT").

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

            in_bc = self.get_u_BC(
                VANO_model=VANO_model, normalizer_u_bc=normalizer_u_bc
            )
            in_f = MultipleTensors([i for i in (in_bc, a0)])

            return (
                self.g,
                torch.cat((self.theta.to(self.device), self.RT, self.CT)),
                in_f,
            )

        else:
            raise ValueError("Model not recognized must be GNOT")

    def log(self):
        """
        Logs the artery parameters to Weights and Biases (wandb).

        This function plots the reconstructed and true boundary conditions (u_bc)
        for an artery and logs the plot to wandb.

        The plot includes:
        - Reconstructed boundary conditions (u_bc_rec)
        - True boundary conditions (u_bc_true)

        The plot is titled with the artery name.

        Note:
            - The function assumes that `self.u_bc_rec` and `self.u_bc_true` are
              PyTorch tensors.
            - The function uses `wandb` for logging.

        """
        plt.plot(self.u_bc_rec.detach().cpu().numpy(), label="reconstructed")
        plt.plot(self.u_bc_true.detach().cpu().numpy(), label="true")
        plt.title(f"U_bc for artery {self.name}")
        plt.legend()
        wandb.log({self.name: plt})
        plt.close()

    def log_mesurement(self):
        """
        Logs the predicted and true values at the measurement point.

        This function plots the predicted values (`u_mesurement`) and the true values
        (`mesurement_value`) at a specific measurement point for an artery. The plot
        is then logged using Weights and Biases (wandb) for comparison.

        Attributes:
            u_mesurement (torch.Tensor): The predicted values at the measurement point.
            mesurement_value (torch.Tensor): The true values at the measurement point.
            name (str): The name of the artery being measured.

        Plots:
            A plot comparing the predicted and true values at the measurement point.

        Logs:
            The plot is logged to Weights and Biases with the key `comparison_{self.name}`.
        """

        plt.plot(self.u_mesurement.detach().cpu().numpy(), label="reconstructed")
        plt.plot(self.mesurement_value.detach().cpu().numpy(), label="true")
        plt.title(f"Mesurement point for artery {self.name}")
        plt.legend()

        wandb.log({f"comparison_{self.name}": plt})

        # plt.show()
        plt.close()

    def dump_mesurement(self, fig_path: str):
        """
        Dumps predicted and true values at the measurement point to a figure.
        Args:
            fig_path (str): The path where the figure will be saved.
        Returns:
            None
        The function creates a plot with the following:
        - Noisy measurement values (blue, dotted line)
        - Ground truth values (black, solid line)
        - Reconstructed values (red, dashed line)
        The plot includes a grid, labels for the x and y axes, a legend, and customized tick sizes.
        The figure is saved as a PNG file at the specified path.
        """

        plt.figure(figsize=(7.48 / 2, 9.45 / 2), dpi=300)
        plt.plot(
            (self.mesurement_value).detach().cpu().numpy(),
            label="Noisy measurement",
            color="blue",
            linestyle=":",
        )
        plt.plot(
            (self.mesurement_value - self.noise).detach().cpu().numpy(),
            label="Ground truth",
            color="black",
        )
        plt.plot(
            self.u_mesurement.detach().cpu().numpy(),
            label="Reconstructed",
            color="red",
            linestyle="--",
        )
        plt.grid()
        # plt.title(f"Measurement and predicted velocity {self.name}")
        plt.ylabel(r"$cm \cdot s^{-1}$", fontsize=7)
        plt.xlabel("Time step", fontsize=7)
        plt.legend(fontsize=6)
        plt.xticks(fontsize=7)
        plt.yticks(fontsize=7)

        plt.savefig(fig_path + f"/comparison_{self.name}_new.png")
        plt.close()

    def set_u_in(self, u_in: torch.Tensor):
        """
        Sets the input tensor `u_in`.

        Args:
            u_in (torch.Tensor): The input tensor to be set.
        """
        self.u_in = u_in

    def get_u_in(self):
        """
        Retrieve the value of the u_in attribute.

        Returns:
            The value of the u_in attribute.
        """
        return self.u_in

    def set_u_out(self, u_out: torch.Tensor):
        """
        Sets the output tensor `u_out`.

        Args:
            u_out (torch.Tensor): The output tensor to be set.
        """
        self.u_out = u_out

    def get_u_out(self):
        """
        Retrieve the value of the u_out attribute.

        Returns:
            The value of the u_out attribute.
        """
        return self.u_out

    def set_a_in(self, a_in: torch.Tensor):
        """
        Sets the input tensor `a_in`.

        Args:
            a_in (torch.Tensor): The input tensor to be set.
        """
        self.a_in = a_in

    def get_a_in(self):
        """
        Retrieve the value of the attribute 'a_in'.

        Returns:
            The value of the 'a_in' attribute.
        """
        return self.a_in

    def set_a_out(self, a_out: torch.Tensor):
        """
        Sets the output tensor `a_out`.

        Args:
            a_out (torch.Tensor): The output tensor to be set.
        """
        self.a_out = a_out

    def get_a_out(self):
        """
        Retrieve the value of the attribute 'a_out'.

        Returns:
            The value of 'a_out'.
        """
        return self.a_out

    def set_p_in(self, p_in: torch.Tensor):
        """
        Sets the input tensor for the operator.

        Args:
            p_in (torch.Tensor): The input tensor to be set.
        """
        self.p_in = p_in

    def get_p_in(self):
        """
        Retrieve the value of the p_in attribute.

        Returns:
            The value of the p_in attribute.
        """
        return self.p_in

    def set_p_out(self, p_out: torch.Tensor):
        """
        Sets the output tensor.

        Args:
            p_out (torch.Tensor): The output tensor to be set.
        """
        self.p_out = p_out

    def get_p_out(self):
        """
        Retrieve the value of the p_out attribute.

        Returns:
            The value of the p_out attribute.
        """
        return self.p_out

    def set_RT(self, RT: torch.Tensor):
        """
        Sets the RT attribute with the provided tensor.

        Args:
            RT (torch.Tensor): The tensor to set as the RT attribute.
        """
        self.RT = RT

    def set_CT(self, CT: torch.Tensor):
        """
        Sets the CT attribute with the provided tensor.

        Args:
            CT (torch.Tensor): The tensor to set as the CT attribute.
        """
        self.CT = CT

    def has_mesurement(self):
        """
        Check if the measurement is available.

        Returns:
            bool: True if the measurement is available, False otherwise.
        """
        return self.mesurement

    def set_reconstructed_u_mesurement(self, u):
        """
        Sets the reconstructed u measurement.

        Parameters:
        u (float): The value of the u measurement to be set.
        """
        self.u_mesurement = u

    def get_reconstructed_u_mesurement(self):
        """
        Retrieve the reconstructed u_measurement.

        Returns:
            The reconstructed u_measurement.
        """
        return self.u_mesurement

    def get_true_mesurement(self):
        """
        Retrieve the true measurement value.

        Returns:
            The measurement value stored in the instance.
        """
        return self.mesurement_value

    def get_a0(self):
        """
        Returns the attribute `a0` after removing single-dimensional entries from its shape.

        Returns:
            numpy.ndarray: The `a0` attribute with single-dimensional entries removed.
        """
        return self.a0.squeeze()

    def set_a0_rec(self, a0_rec):
        """
        Sets the value of the a0_rec attribute.

        Parameters:
        a0_rec (type): The value to set for the a0_rec attribute.
        """
        self.a0_rec = a0_rec

    def get_a0_rec(self):
        """
        Retrieve the value of the a0_rec attribute.

        Returns:
            The value of the a0_rec attribute.
        """
        return self.a0_rec

    def get_true_u_in(self):
        """
        Retrieve the true input velocity (u_in).

        Returns:
            float: The true input velocity.
        """
        return self.u_in_true

    def set_true_u_in(self, u_in_true):
        """
        Sets the true input velocity.

        Parameters:
        u_in_true (float): The true input velocity to be set.
        """
        self.u_in_true = u_in_true

    def get_true_a_in(self):
        """
        Retrieve the true value of 'a_in'.

        Returns:
            The true value of 'a_in'.
        """
        return self.a_in_true

    def set_true_a_in(self, a_in_true):
        """
        Sets the true value of 'a_in'.

        Parameters:
        a_in_true (type): The true value to be assigned to 'a_in_true'.
        """
        self.a_in_true = a_in_true

    def get_true_p_in(self):
        """
        Retrieve the true input pressure value.

        Returns:
            float: The true input pressure value.
        """
        return self.p_in_true

    def set_true_p_in(self, p_in_true):
        """
        Sets the true input pressure value.

        Parameters:
        p_in_true (float): The true input pressure value to be set.
        """
        self.p_in_true = p_in_true

    def is_root(self):
        """
        Check if the current node is the root.

        Returns:
            bool: True if the current node is the root, False otherwise.
        """
        return self.root

    def get_T(self):
        """
        Returns the value of the attribute T.

        Returns:
            The value of the attribute T.
        """
        return self.T

    def get_r0(self):
        """
        Calculate and return the value of r0.

        This method computes r0 by taking the square root of the first element of
        the tensor `a0` divided by π. The result is then detached from the computation
        graph, moved to the CPU, and converted to a NumPy array.

        Returns:
            numpy.ndarray: The computed value of r0 as a NumPy array.
        """
        return torch.sqrt(self.a0[0] / torch.pi).detach().cpu().numpy()

    def get_t(self):
        """
        Retrieves the first 100 elements of the second column from the node data.

        Returns:
            numpy.ndarray: An array containing the first 100 elements of the second column of the node data.
        """
        return self.g.ndata["x"][:100, 1]

    def get_L(self):
        """
        Retrieves the tensor `L` as a NumPy array.

        This method detaches the tensor `L` from the current computation graph,
        moves it to the CPU, and converts it to a NumPy array.

        Returns:
            numpy.ndarray: The tensor `L` as a NumPy array.
        """
        return self.L.detach().cpu().numpy()

    def is_outlet(self):
        """
        Check if the current object is an outlet.

        This method determines if the `name` attribute of the current object
        matches any of the predefined outlet names.

        Returns:
            bool: True if `name` is one of the outlet names, False otherwise.
        """
        return self.name in [
            "L_PCA_P2",
            "R_PCA_P2",
            "L_ACA_A2",
            "R_ACA_A2",
            "L_MCA",
            "R_MCA",
        ]

    def set_p_pred(self, pred: torch.Tensor):
        """
        Sets the predicted tensor.

        Args:
            pred (torch.Tensor): The predicted tensor to be set.
        """
        self.p_pred = pred

    def get_p_pred(self):
        """
        Retrieve the predicted pressure value.

        Returns:
            float: The predicted pressure value.
        """
        return self.p_pred

    def get_u_bc_rec(self):
        """
        Retrieve the boundary condition reconstruction.

        Returns:
            The boundary condition reconstruction (u_bc_rec).
        """
        return self.u_bc_rec

    def get_latent(self):
        """
        Retrieve the latent boundary condition.

        Returns:
            The latent boundary condition (u_bc_latent).
        """
        return self.u_bc_latent


class COW(object):
    """
    Class for representing circle of willis model and solving inverse problem associated with it.
    COntains all methods for reconstructing input parametres
    Attributes
    rho : float
        Density in CGS units.
    RT_true : float or None
        True RT value.
    CT_true : float or None
        True CT value.
    HR : float or None
        Heart rate.
    lr : float
        Learning rate.
    RT : torch.Tensor
        RT tensor with gradient enabled.
    CT : torch.Tensor
        CT tensor derived from RT.
    VANO : bool
        VANO flag.
    device : str
        Device to run the model on.
    model_surrogate : nn.Module
        Surrogate model.
    model_VANO : nn.Module
        VANO model.
    track : bool
        Tracking flag.
    normalizer_x : Normalizer
        Normalizer for x.
    normalizer_y : Normalizer
        Normalizer for y.
    normalizer_theta : Normalizer
        Normalizer for theta.
    normalizer_u_bc : Normalizer or None
        Normalizer for u_bc.
    joints_path : str
        Path to joints CSV file.
    l2_loss : WeightedLpRelLoss
        L2 loss function.
    optimizer_mes : torch.optim.Optimizer or None
        Optimizer for measured arteries.
    optimizer_non_mes : torch.optim.Optimizer or None
        Optimizer for non-measured arteries.
    optimizer_full : torch.optim.Optimizer or None
        Optimizer for all arteries.
    optimizer_RT : torch.optim.Optimizer or None
        Optimizer for RT.
    p_RT : list
        Parameters for RT optimizer.
    p_MES : list
        Parameters for measured arteries optimizer.
    p_NON_MES : list
        Parameters for non-measured arteries optimizer.
    p_FULL : list
        Parameters for full optimizer.
    arteries : list
        List of artery objects.
    joints : list
        List of joint tuples.
    Methods
    -------
    __init__(self, model_surrogate, data_path, track, device, normalizer_x, normalizer_y, normalizer_theta, joints_path, lr, VANO, model_VANO, normalizer_u_bc=None, idx=1)
        Initialize the COW model.
    load_data(self, data_path)
        Load data from specified path.
    create_joints(self, joints_csv)
        Create joints from CSV file.
    create_optimizer(self, lr, optim)
        Create optimizer for arterial parameters.
    loader_GNOT(self, batch_size, batch_idx=None)
        Load GNOT data in batches.
    solve_arteries(self, batch_size, batch_idx=None)
        Compute solution for arteries.
    compute_validation_l2_loss(self, batch_size)
        Compute validation L2 loss.
    compute_bifurcation_loss(self, j=None)
        Compute bifurcation loss.
    compute_mesurement_loss(self, batch_idx=None)
        Compute measurement loss.
    update_arteries(self, pred, idx)
    get_arteries(self, idx, model='GNOT')
    dump_solutions(self, path, pred, idx)
        Save predictions for each artery.
    mass_conservation(self)
        Penalize difference between mass entering and leaving COW.
    propagate_RT(self)
        Pass RT value to arteries.
    update_CT(self)
        Update CT value.
    propagate_CT(self)
        Pass CT value to arteries.
    log_arteries(self)
        Log all arteries to wandb.
    compute_a0_loss(self, batch_idx=None)
        Compute a0 loss.
    solve_accumulate_2(self, max_iters, eps, batch_size, lambda_mes, lambda_mass, lambda_pressure, lambda_a0, run_id=0)
        Deprecated function for solving and accumulating losses.
    log_validation(self)
        Log validation statistics to wandb.
    get_validation(self, arteries)
        Compute validation statistics for plotting.
    dump_plots(self, path)
        Create and save plots of u, a, p, and flow for each artery.
    dump_mesurement_plots(self, path)
        Create and save plots of measurements.
    create_subsegments(self, a, b, n)
    sample_RT(self, idx)
    dump_params(self, path)
        Save simulation parameters to a text file.
    dump_statistics(self, path)
        Save statistics of predicted and true values of u, a, p to a text file.
    dump_validation(self, path, arteries)
        Save validation statistics to a text file.
    dump_reconstructed_u_bc_plots(self, path)
        Save reconstructed u values to a path.
    get_num_arteries(self)
        Return the number of arteries in COW.
    get_artery(self, idx)
        Return artery with specified index.
    get_r0s(self)
        Return r0 values for all arteries.
    get_Ls(self)
        Return lengths of all arteries.
    get_outlet_predictions(self, true=False)
        Return outlet predictions.
    dump_RT(self, path)
        Save RT to a specified path.
    """

    def __init__(
        self,
        model_surrogate: nn.Module,
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
        idx: int = 1,
    ):
        self.rho = 1.06  ## must be in CGS units
        self.RT_true = None
        self.CT_true = None
        self.HR = None
        self.lr = lr
        torch.manual_seed(2137)
        self.RT = torch.Tensor([self.sample_RT(idx)]).to(device).requires_grad_(True)

        self.CT = 1.34 / self.RT.detach()
        self.VANO = VANO
        self.device = device
        self.model_surrogate = model_surrogate
        self.model_VANO = model_VANO
        self.track = track
        self.normalizer_x = normalizer_x
        self.normalizer_y = normalizer_y
        if normalizer_u_bc is not None:
            self.normalizer_u_bc = normalizer_u_bc
        self.normalizer_theta = normalizer_theta
        self.joints_path = joints_path  # TODO
        self.l2_loss = WeightedLpRelLoss(p=2, component="all", normalizer=None)
        self.load_data(data_path=data_path)
        self.optimizer_mes = None
        self.optimizer_non_mes = None
        self.optimizer_full = None
        self.create_optimizer(lr, "RT")
        self.propagate_CT()
        self.propagate_RT()
        self.joints = self.create_joints(joints_path)

    def load_data(
        self,
        data_path: str,
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

        for artery in arteries:
            X, Y, theta, in_funcs = np.load(
                data_path + artery + ".npy", allow_pickle=True
            )

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
            g.ndata["y"] = torch.from_numpy(Y).float()

            theta = torch.from_numpy(theta).float()
            condition = self.normalizer_theta.transform(
                theta.to(self.device), inverse=False
            )
            condition = condition[:, :11]
            # passing also true values for comparison
            input_f = [torch.from_numpy(in_func).float() for in_func in in_funcs]

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
                )

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

    def create_optimizer(self, lr: float = 0.5, optim: str = "RT"):
        """
        Creates an optimizer for arterial parameters based on the specified optimization type.

        Parameters:
        lr (float): Learning rate for the optimizer. Default is 0.5.
        optim (str): Type of optimizer to create. Must be one of ["RT", "MES", "NON_MES", "FULL"].
                     - "RT": Optimizer for RT parameter.
                     - "MES": Optimizer for parameters of arteries with measurements.
                     - "NON_MES": Optimizer for parameters of arteries without measurements.
                     - "FULL": Optimizer for all arterial parameters.

        Raises:
        ValueError: If the specified optimizer type is not recognized.

        Sets:
        self.optimizer_RT: Adam optimizer for RT parameter if optim is "RT".
        self.p_RT: List of parameters for RT optimizer if optim is "RT".
        self.optimizer_mes: Adam optimizer for arteries with measurements if optim is "MES".
        self.p_MES: List of parameters for MES optimizer if optim is "MES".
        self.optimizer_non_mes: Adam optimizer for arteries without measurements if optim is "NON_MES".
        self.p_NON_MES: List of parameters for NON_MES optimizer if optim is "NON_MES".
        self.optimizer_full: Adam optimizer for all arterial parameters if optim is "FULL".
        self.p_FULL: List of parameters for FULL optimizer if optim is "FULL".
        """

        if optim not in ["RT", "MES", "NON_MES", "FULL"]:
            raise ValueError("Optimizer not recognized")

        if optim == "RT":
            p = [self.RT.requires_grad_(True)]
            self.optimizer_RT = torch.optim.Adam(p, lr=lr)
            self.p_RT = p

        elif optim == "MES":
            p = []

            for artery in self.arteries:
                if artery.get_parameters() is not None and artery.has_mesurement():
                    p.extend(artery.get_parameters())
            p.extend([self.RT.requires_grad_(True)])
            self.optimizer_mes = torch.optim.Adam(p, lr=lr)
            self.p_MES = p

        elif optim == "NON_MES":
            p = []
            for artery in self.arteries:
                if artery.get_parameters() is not None and not artery.has_mesurement():
                    p.extend(artery.get_parameters())
            p.extend([self.RT.requires_grad_(True)])
            self.optimizer_non_mes = torch.optim.Adam(p, lr=lr)
            self.p_NON_MES = p

        elif optim == "FULL":
            p = []
            for artery in self.arteries:
                if artery.get_parameters() is not None:
                    p.extend(artery.get_parameters())
            p.extend([self.RT.requires_grad_(True)])
            self.optimizer_full = torch.optim.Adam(p, lr=lr)
            self.p_FULL = p

    def loader_GNOT(self, batch_size, batch_idx=None):
        """
        Loads batches of data for the GNOT model.

        Args:
            batch_size (int): The size of each batch.
            batch_idx (list, optional): Specific indices for batching. If None, batches are created from the entire dataset.

        Yields:
            tuple: A tuple containing:
                - batched (list): A list of batched data, where each element is either a batched DGLGraph, a stacked torch.Tensor, or a MultipleTensors object.
                - indices (list): The indices of the data in the current batch.

        Raises:
            NotImplementedError: If the data type of the sample is not supported.
        """
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
        Computes the solution for single artery batches.
        This function processes batches of artery data, applies transformations,
        and updates the artery states based on the model's surrogate predictions.
        Parameters:
        batch_size (int): The size of each batch to process.
        batch_idx (optional): Specific indices of batches to process. If None, all batches are processed.
        Returns:
        tuple: A tuple containing the output tensor and the corresponding indices.
        """

        if batch_idx is not None:
            batch_idx = batch_idx

            for batch, idx in self.loader_GNOT(batch_size, batch_idx):
                g, u_p, g_u = batch

                g, u_p, g_u = (
                    g.to(self.device),
                    u_p.to(self.device),
                    g_u.to(self.device),
                )

                g.ndata["x"] = self.normalizer_x.transform(g.ndata["x"], inverse=False)
                u_p = self.normalizer_theta.transform(u_p, inverse=False)
                out = self.model_surrogate(
                    g, u_p, g_u
                )  # trzeba zrobic reshape bo jest [bs * n_nodes, 3]

                out = self.normalizer_y.transform(out, inverse=True)
                out = out.reshape(len(batch_idx), -1, 3)

                self.update_arteries(out, idx)
                return out, idx

        else:
            for batch, idx in self.loader_GNOT(batch_size):

                g, u_p, g_u = batch

                g, u_p, g_u = (
                    g.to(self.device),
                    u_p.to(self.device),
                    g_u.to(self.device),
                )

                g.ndata["x"] = self.normalizer_x.transform(g.ndata["x"], inverse=False)
                u_p = self.normalizer_theta.transform(u_p, inverse=False)

                out = self.model_surrogate(g, u_p, g_u)
                out = self.normalizer_y.transform(out, inverse=True)
                out = out.reshape(batch_size, -1, 3)

                self.update_arteries(out, idx)
                return out, idx

    def compute_validation_l2_loss(self, batch_size: int):
        """
        Computes the validation L2 loss for all arteries.
        Args:
            batch_size (int): The size of the batch to be processed.
        Returns:
            float: The average L2 loss over all batches.
        Note:
            This function uses a DataLoader to iterate over the validation dataset in batches.
            It computes the L2 loss for each batch and accumulates the loss to compute the average.
            The function operates in no_grad mode to avoid computing gradients during validation.
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

        return loss / i

    def compute_bifurcation_loss(self, j=None):
        """
        Computes the bifurcation loss for the given joint or all joints if no specific joint is provided.
        The bifurcation loss is calculated based on the conservation of mass and total pressure continuity.
        Parameters:
        j (tuple, optional): A tuple representing a specific joint in the format (p, d1, d2, merging), where:
            - p: Index of the parent artery.
            - d1: Index of the first daughter artery.
            - d2: Index of the second daughter artery.
            - merging: Boolean indicating whether the joint is merging (True) or splitting (False).
        Returns:
        tuple: A tuple containing:
            - loss_mass (torch.Tensor): The computed mass loss.
            - loss_pressure (torch.Tensor): The computed pressure loss.
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

                    p1 = p.get_p_out()
                    pd1 = d1.get_p_in()
                    pd2 = d2.get_p_in()

                    loss_pressure += torch.mean(torch.square(p1 - pd2)) + torch.mean(
                        torch.square(p1 - pd1)
                    )
        return loss_mass, loss_pressure

    def compute_mesurement_loss(self, batch_idx=None):
        """
        Computes the measurement loss for the given batch of arteries or for all arteries if no batch is specified.

        Parameters:
        batch_idx (list, optional): List of indices specifying which arteries to include in the loss computation.
                                    If None, computes the loss for all arteries.

        Returns:
        float: The total measurement loss computed as the sum of mean squared errors between the reconstructed
               and true measurements for the specified arteries.
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

    def update_arteries(self, pred: torch.Tensor, idx: List[int]):
        """
        Update arteries with new predictions.

        This function updates the attributes of arteries with new prediction values.
        The predictions are provided in the form of a tensor, and the indices of the
        arteries to be updated are provided in a list.

        Args:
            pred (torch.Tensor): A tensor containing the prediction values. The tensor
                should have the following structure:
                [
                    [[p(x0, t0), a(x0, t0), u(x0, t0)],
                    [p(x0, t1), a(x0, t1), u(x0, t1)],
                    ...
                where p, a, and u represent pressure, area, and velocity respectively.
            idx (List[int]): A list of indices indicating which arteries to update.

        The function updates the following attributes of each artery:
            - set_u_in: Sets the input velocity using the first 100 values of the third column.
            - set_u_out: Sets the output velocity using the last 100 values of the third column.
            - set_a_in: Sets the input area using the first 100 values of the second column.
            - set_a_out: Sets the output area using the last 100 values of the second column.
            - set_p_in: Sets the input pressure using the first 100 values of the first column.
            - set_p_out: Sets the output pressure using the last 100 values of the first column.
            - set_a0_rec: Sets the reconstructed area using every 100th value of the second column.
            - set_p_pred: Sets the predicted pressure using all values of the first column.
            - set_reconstructed_u_mesurement: Sets the reconstructed velocity measurement using
              values from index 2500 to 2600 of the third column, if the artery has a measurement.

        Note:
            The function assumes that the pred tensor has at least 2600 rows to avoid index errors.
        """

        for i, idx in enumerate(idx):
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

    def get_arteries(self, idx, model: str = "GNOT"):
        """
        Retrieve artery inputs for a given index and model.

        Parameters:
        idx (int): The index of the artery to retrieve.
        model (str, optional): The model type to use for retrieving inputs. Defaults to "GNOT".

        Returns:
        Any: The inputs for the specified artery and model.

        Raises:
        ValueError: If the artery inputs cannot be retrieved.
        """
        try:
            return self.arteries[idx].get_inputs(
                model, self.model_VANO, self.normalizer_u_bc
            )
        except ValueError:
            pass

    def dump_solutions(self, path: str, pred: torch.Tensor, idx: List[int]):
        """
        Saves prediction results for pressure, velocity, and area for each artery to .npy files.

        Args:
            path (str): Directory path where the solution files will be saved
            pred (torch.Tensor): Tensor containing predictions with shape (n_arteries, timesteps, 3)
                                where the last dimension contains [pressure, area, velocity]
            idx (List[int]): List of artery indices to process

        Notes:
            - Files are saved in the format: {artery_name}_{quantity}.npy
            - Quantities saved are 'velocity', 'pressure', and 'area'
            - Predictions are detached from computational graph and converted to numpy arrays
        """

        for i, idx in enumerate(idx):
            # save velocity
            np.save(
                path + "/" + self.arteries[idx].name + "_velocity" + ".npy",
                pred[i, :, 2].detach().cpu().numpy(),
            )
            # save pressure
            np.save(
                path + "/" + self.arteries[idx].name + "_pressure" + ".npy",
                pred[i, :, 0].detach().cpu().numpy(),
            )
            # save area
            np.save(
                path + "/" + self.arteries[idx].name + "_area" + ".npy",
                pred[i, :, 1].detach().cpu().numpy(),
            )

    def mass_conservation(self):
        """
        Computes the mass conservation penalty for the cow model.

        This function calculates the difference between the mass flow rate
        entering and leaving the cow model. It iterates through the arteries
        to sum up the inflow and outflow rates. The penalty is the mean
        squared difference between the total inflow and outflow rates.

        Returns:
            torch.Tensor: The mean squared difference between the inflow
            and outflow rates.
        """

        Q_in = None
        Q_out = None
        for artery in self.arteries:
            if artery.is_root():
                if Q_in is None:
                    Q_in = artery.get_u_in() * artery.get_a_in()
                else:
                    Q_in += artery.get_u_in() * artery.get_a_in()
            elif artery.is_outlet():
                if Q_out is None:
                    Q_out = artery.get_u_out() * artery.get_a_out()
                else:
                    Q_out += artery.get_u_out() * artery.get_a_out()
        return torch.mean(torch.square(Q_in - Q_out))

    def propagate_RT(self):
        """
        Propagates the RT value to all arteries.

        This method iterates over all arteries in the `self.arteries` list and sets
        their RT value to the RT value of the current object.

        Returns:
            None
        """

        for artery in self.arteries:
            artery.set_RT(self.RT)

    def update_CT(self):
        """
        Updates the CT value by calculating the ratio of a constant to the detached RT value.

        This method updates the `CT` attribute of the instance by dividing the constant 1.34 by the
        detached value of the `RT` attribute.

        Returns:
            None
        """

        self.CT = 1.34 / self.RT.detach()

    def propagate_CT(self):
        """
        Propagates the CT value to all arteries.

        This method iterates over all arteries in the `arteries` attribute and sets
        their CT value to the current CT value of the instance.

        Attributes:
            arteries (list): A list of artery objects that have a `set_CT` method.
            CT (float): The CT value to be propagated to the arteries.

        """

        for artery in self.arteries:
            artery.set_CT(self.CT)

    def log_arteries(self):
        """
        Logs all arteries to Weights and Biases (wandb).
        This function iterates through the list of arteries and logs each one using
        their respective log method.
        Returns:
            None
        """

        for artery in self.arteries:
            artery.log()

    def compute_a0_loss(self, batch_idx=None):
        """
        Compute the mean squared error (MSE) loss for the a0 parameter of arteries.
        This method calculates the MSE loss between the reconstructed a0 values and the
        actual a0 values of the arteries. If a batch index is provided, the loss is
        computed only for the specified batch indices. Otherwise, the loss is computed
        for all arteries.
        Args:
            batch_idx (list, optional): A list of indices specifying which arteries to
                                        include in the loss computation. If None, the
                                        loss is computed for all arteries.
        Returns:
            float: The computed MSE loss.
        """
        loss = 0
        if batch_idx is not None:
            for idx in batch_idx:
                loss += nn.MSELoss()(
                    self.arteries[idx].get_a0_rec(), self.arteries[idx].get_a0()
                )
            return loss
        else:
            for artery in self.arteries:

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
        Deprecated
        """
        raise NotImplementedError("Function deprecated use Multiple COW function")

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

        wandb.log({"Validation loss": validation_loss})

        return validation_loss, loss

    def log_validation(self):
        """
        Logs the validation metrics for each artery using the Weights and Biases (wandb) library.

        This method iterates over the arteries and their corresponding data batches, performs
        normalization, runs the surrogate model to get predictions, computes the L2 loss for
        area, pressure, velocity, and flow, and logs these metrics in a wandb table.

        The table contains the following columns:
        - "Artery": Name of the artery
        - "rL2 Area": Relative L2 loss for the area
        - "rL2 Pressure": Relative L2 loss for the pressure
        - "rL2 Velocity": Relative L2 loss for the velocity
        - "rL2 Flow": Relative L2 loss for the flow

        The method uses torch.no_grad() to ensure no gradients are computed during validation.

        Args:
            None

        Returns:
            None
        """
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
        Computes validation statistics for plotting.

        Args:
            arteries (dict): A dictionary where keys are artery names and values are dictionaries
                             containing lists to store validation statistics for 'Area', 'Pressure',
                             'Velocity', and 'Flow'.

        Returns:
            dict: Updated arteries dictionary with computed validation statistics.

        The function performs the following steps:
        1. Disables gradient calculation for validation.
        2. Iterates over each artery and its corresponding data loader.
        3. Normalizes input features and parameters.
        4. Passes the normalized data through the surrogate model to get predictions.
        5. Transforms the predictions back to the original scale.
        6. Computes the L2 loss for 'Area', 'Pressure', 'Velocity', and 'Flow'.
        7. Appends the computed losses to the corresponding lists in the arteries dictionary.
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

    def dump_plots(self, path: str):
        """
        function creates plots of u, a, p  and flow for each artery
        with true and predicted values at inlet with Title of artery name
        and saves them to path with name artery_name.png

        function makes use of artery.get_u_in(), artery.get_a_in(), artery.get_p_in()
        """
        name_map = {
            "L_PCA_P1": "Left PCA P1",
            "L_PCA_P2": "Left PCA P2",
            "R_PCA_P1": "Right PCA P1",
            "R_PCA_P2": "Right PCA P2",
            "L_MCA": "Left MCA",
            "R_MCA": "Right MCA",
            "L_ACA_A1": "Left ACA A1",
            "L_ACA_A2": "Left ACA A2",
            "R_ACA_A1": "Right ACA A1",
            "R_ACA_A2": "Right ACA A2",
            "AcoA": "ACoA",
            "L_int_carotid_I": "Left Internal Carotid I",
            "L_int_carotid_II": "Left Internal Carotid II",
            "R_int_carotid_I": "Right Internal Carotid I",
            "R_int_carotid_II": "Right Internal Carotid II",
            "L_PcoA": "Left PCoA",
            "R_PcoA": "Right PCoA",
            "Basilar": "Basilar",
        }
        for artery in self.arteries:
            u_in = artery.get_u_in().detach().cpu().numpy()
            a_in = artery.get_a_in().detach().cpu().numpy()
            p_in = artery.get_p_in().detach().cpu().numpy()
            u_in_true = artery.get_true_u_in().detach().cpu().numpy()
            a_in_true = artery.get_true_a_in().detach().cpu().numpy()
            p_in_true = artery.get_true_p_in().detach().cpu().numpy()

            fig, axs = plt.subplots(2, 2, figsize=(10, 10))
            fig.suptitle(f"Reconstructed vs Ground Truth {name_map[artery.name]}")

            axs[0, 0].plot(unit_to_mmHg(p_in_true), label="Ground truth", color="black")
            axs[0, 0].plot(
                unit_to_mmHg(p_in), label="Predicted", color="red", linestyle="--"
            )
            axs[0, 0].set_title("Pressure")
            axs[0, 0].set_ylabel("mmHg")
            axs[0, 0].set_xlabel("Time step")
            axs[0, 0].legend()
            axs[0, 0].grid()

            axs[0, 1].plot(a_in_true, label="Ground truth", color="black")
            axs[0, 1].plot(a_in, label="Predicted", color="red", linestyle="--")
            axs[0, 1].set_title("Area")
            axs[0, 1].set_ylabel("cm^2")
            axs[0, 1].set_xlabel("Time step")
            org_lim = axs[0, 1].get_ylim()
            axs[0, 1].set_ylim(
                [org_lim[0] - org_lim[0] * 0.10, org_lim[1] + org_lim[1] * 0.10]
            )
            axs[0, 1].legend()
            axs[0, 1].grid()

            axs[1, 0].plot(u_in_true, label="Ground truth", color="black")
            axs[1, 0].plot(u_in, label="Predicted", color="red", linestyle="--")

            axs[1, 0].set_title("Velocity")
            axs[1, 0].set_ylabel("cm/s")
            axs[1, 0].set_xlabel("Time step")
            axs[1, 0].legend()
            axs[1, 0].grid()
            axs[1, 1].plot(u_in_true * a_in_true, label="Ground truth", color="black")
            axs[1, 1].plot(u_in * a_in, label="Predicted", color="red", linestyle="--")

            axs[1, 1].set_title("Flow")
            axs[1, 1].set_ylabel("cm^3/s")
            axs[1, 1].set_xlabel("Time step")
            axs[1, 1].legend()
            axs[1, 1].grid()

            plt.savefig(os.path.join(path, f"{artery.name}.png"))
            plt.close()

    def dump_mesurement_plots(self, path: str):
        """
        Creates and saves plots of measurements for each artery.

        Parameters:
        path (str): The directory path where the plots will be saved.

        This function iterates over all arteries and checks if they have measurements.
        If an artery has measurements, it calls the artery's dump_mesurement method
        to save the plots to the specified path.
        """

        for artery in self.arteries:
            if artery.has_mesurement():
                artery.dump_mesurement(path)

    def create_subsegments(
        self, a: float, b: float, n: int
    ) -> List[Tuple[float, float]]:
        """
        Subdivide segment [a, b] into n disjoint subsegments.

        Args:
            a (float): Left endpoint of the segment
            b (float): Right endpoint of the segment
            n (int): Number of subsegments

        Returns:
            List[Tuple[float, float]]: List of subsegments represented as tuples (left, right)

        Raises:
            ValueError: If n < 1 or a >= b
        """
        if n < 1:
            raise ValueError("Number of subsegments must be positive")
        if a >= b:
            raise ValueError("Left endpoint must be less than right endpoint")

        # Calculate the length of each subsegment
        segment_length = (b - a) / n

        # Create subsegments
        subsegments = []
        for i in range(n):
            left = a + i * segment_length
            right = a + (i + 1) * segment_length
            subsegments.append((left, right))

        return subsegments

    def sample_RT(self, idx: int) -> float:
        """
        Sample RT value for the idx-th artery.

        Args:
            idx (int): Index of the artery

        Returns:
            float: Sampled RT value
        """
        subsegments = self.create_subsegments(62770839.96, 194904651.43, 10)
        return np.random.uniform(subsegments[idx][0], subsegments[idx][1])

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
            u_in_true = artery.u_bc_true.detach().cpu().numpy()

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
        Returns a dictionary with outlet predictions for arteries.

        Parameters:
        true (bool): If True, returns true inlet values transformed to SI units.
                     If False, returns outlet values transformed to SI units. Default is False.

        Returns:
        dict: A dictionary where the keys are the indices of the arteries and the values are tuples containing:
              - Artery area (in square meters)
              - Artery velocity (in meters per second)
              - Artery pressure (in kilopascals)
              - Time (in seconds)
        """

        out = dict()
        for idx, artery in enumerate(self.arteries):
            if artery.is_outlet():
                # need to transfrom to si units
                if true:
                    out[idx] = (
                        artery.get_true_u_in().detach().cpu().numpy() * 1e-2,
                        artery.get_true_a_in().detach().cpu().numpy() * 1e-4,
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

    def dump_RT(self, path: str):
        """
        Function dumps RT to path
        """
        np.save(path + "_RT" + ".npy", self.RT.detach().cpu().numpy())


class Multiple_COWs(object):
    """
    Super class for executing multiple instances of COW in parallel.
    This class provides methods to initialize, solve, and optimize multiple instances of COW (Circle of Willis) models in parallel. It includes functionality for batching, normalization, and optimization of the models.
    Attributes:
        COWs (List[COW]): List of COW instances.
        device (torch.device): Device to run the computations on (CPU or GPU).
        normalizer_x: Normalizer for input data x.
        normalizer_y: Normalizer for output data y.
        normalizer_theta: Normalizer for parameter theta.
        model_surrogate: Surrogate model for predictions.
        lr: Learning rate for optimizers.
    Methods:
        initialize_losses():
            Initializes the loss and validation lists for each COW instance.
        set_g():
            Sets the graph data for the first batch of COW instances.
        loader(batch_size: int):
            Generator that yields batches of data for multiple COW instances.
        solve_cows(batch=None, idx=None):
            Solves the COW instances for a given batch of data.
        solve_inverse(max_iters: int, eps: float, batch_size: int, lambda_mes: float, lambda_mass: float, lambda_pressure: float, lambda_a0: float, run_id=0):
            Optimization loop for multiple COW instances.
        dump_solutions(path: str):
            Dumps the solutions of the best COW instance to the specified path.
        get_L2():
            Returns the L2 validation loss of the best COW instance.
        dump_plots(path: str, best: bool = True):
            Dumps plots of the best or all COW instances to the specified path.
        dump_params(path: str, best: bool = True):
            Dumps parameters of the best or all COW instances to the specified path.
        dump_statistics(path: str, best: bool = True):
            Dumps statistics of the best or all COW instances to the specified path.
        dump_validation(path: str, arteries, best: bool = True):
            Dumps validation results of the best or all COW instances to the specified path.
        dump_reconstructed_u_bc_plots(path: str, best: bool = True):
            Dumps reconstructed boundary condition plots of the best or all COW instances to the specified path.
        get_validation(arteries_log: dict, best: bool = True):
            Returns the validation results of the best COW instance.
        dump_RT(path: str, best: bool = True):
            Dumps RT data of the best or all COW instances to the specified path.
        dump_mesurement_plots(path):
            Dumps measurement plots of the best COW instance.
        get_best():
            Returns the best COW instance.
    """

    def __init__(
        self,
        COWs: List[COW],
        normalizer_x,
        normalizer_y,
        normalizer_theta,
        model_surrogate,
        lr,
    ):
        self.COWs = COWs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.normalizer_x = normalizer_x
        self.normalizer_theta = normalizer_theta
        self.model_surrogate = model_surrogate
        self.normalizer_y = normalizer_y
        self.lr = lr
        self.set_g()

    def initialize_losses(self):
        """
        Initializes the losses and validation attributes.

        This method sets up the `losses` attribute as a list of zeros with the same length as the `COWs` attribute.
        It also initializes the `validation` attribute as an empty list.
        """
        self.losses = list(0 for i in range(len(self.COWs)))
        self.validation = list()

    def set_g(self):
        """
        Sets the graph `g` attribute for the object by loading a batch of data from the loader.

        This method performs the following steps:
        1. Loads a batch of data using the loader with a batch size of 2.
        2. Extracts the graph `g`, `u_p`, and `g_u` from the batch.
        3. Moves the graph `g` to the specified device.
        4. Transforms the node data `x` of the graph `g` using the normalizer.
        5. Sets the transformed graph `g` as an attribute of the object.

        Note:
            This method breaks after processing the first batch, so only the first batch is used to set the graph `g`.

        """
        for batch, idx in self.loader(2):
            g, u_p, g_u = batch
            g = g.to(self.device)
            g.ndata["x"] = self.normalizer_x.transform(g.ndata["x"], inverse=False)
            self.g = g
            break

    def loader(self, batch_size: int):
        """
        Loads batches of data from the COWs dataset.
        Args:
            batch_size (int): The number of samples per batch.
        Yields:
            Tuple[List[Union[dgl.DGLGraph, torch.Tensor, MultipleTensors]], List[int]]:
                A tuple containing:
                - A list of batched data, where each element is either a batched DGLGraph,
                a stacked torch.Tensor, or a MultipleTensors object with padded sequences.
                - A list of indices corresponding to the COWs in the current batch.
        Raises:
            NotImplementedError: If the sample type is not supported.
        """

        a_idx = list(range(18))
        cow_idx = [
            list(range(i, min(i + batch_size, len(self.COWs))))
            for i in range(0, len(self.COWs), batch_size)
        ]
        for indices in cow_idx:
            transposed = zip(
                *[
                    self.COWs[idx_cow].get_arteries(idx_a)
                    for idx_cow in indices
                    for idx_a in a_idx
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
            yield batched, indices

    def solve_cows(self, batch=None, idx=None):
        """
        Solves the forward problem for a batch of data.
        Parameters:
        batch (tuple, optional): A tuple containing the graph `g`, parameters `u_p`, and ground truth `g_u`.
                                 If None, data will be loaded in batches using `self.loader`.
        idx (list, optional): A list of indices corresponding to the COWs in the batch.
                              If None, indices will be provided by `self.loader`.
        Returns:
        tuple: A tuple containing:
            - out (torch.Tensor): The output tensor after processing through the model and normalizers.
            - idx (list): The list of indices corresponding to the COWs in the batch.
        """
        batch_size = 2
        if batch is None:
            for batch, idx in self.loader(batch_size=batch_size):
                g, u_p, g_u = batch

                g, u_p, g_u = (
                    g.to(self.device),
                    u_p.to(self.device),
                    g_u.to(self.device),
                )

                g.ndata["x"] = self.normalizer_x.transform(g.ndata["x"], inverse=False)
                u_p = self.normalizer_theta.transform(u_p, inverse=False)

                out = self.model_surrogate(g, u_p, g_u)
                out = self.normalizer_y.transform(out, inverse=True)
                out = out.reshape(batch_size, 18, -1, 3)

                for idx_out, idx_cow in enumerate(idx):
                    self.COWs[idx_cow].update_arteries(
                        out[idx_out].squeeze(), list(range(18))
                    )

            return out, idx
        else:
            g, u_p, g_u = batch

            u_p, g_u = (
                u_p.to(self.device),
                g_u.to(self.device),
            )

            u_p = self.normalizer_theta.transform(u_p, inverse=False)

            out = self.model_surrogate(self.g, u_p, g_u)
            out = self.normalizer_y.transform(out, inverse=True)
            out = out.reshape(batch_size, 18, -1, 3)

            for idx_out, idx_cow in enumerate(idx):
                self.COWs[idx_cow].update_arteries(
                    out[idx_out].squeeze(), list(range(18))
                )
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
        Solves the inverse problem for the COW model using iterative optimization.
        Parameters:
        -----------
        max_iters : int
            Maximum number of iterations for the optimization process.
        eps : float
            Convergence threshold for the optimization.
        batch_size : int
            Size of the batch for each iteration.
        lambda_mes : float
            Weight for the measurement loss term.
        lambda_mass : float
            Weight for the mass loss term.
        lambda_pressure : float
            Weight for the pressure loss term.
        lambda_a0 : float
            Weight for the a0 loss term.
        run_id : int, optional
            Identifier for the current run (default is 0).
        Returns:
        --------
        None
        Notes:
        ------
        This method initializes the losses and iteratively updates the model parameters
        using different optimizers based on the current iteration. It also computes and
        prints the validation and training losses for each cow in the model.
        """
        it = 0
        self.initialize_losses()
        for i in range(int(max_iters)):

            for batch, idx in self.loader(batch_size):

                _, _ = self.solve_cows(batch, idx)

                for idx_cow in idx:

                    self.losses[idx_cow] = 0
                    if self.COWs[idx_cow].optimizer_full is not None:
                        self.COWs[idx_cow].optimizer_full.zero_grad()
                    if self.COWs[idx_cow].optimizer_non_mes is not None:
                        self.COWs[idx_cow].optimizer_non_mes.zero_grad()
                    if self.COWs[idx_cow].optimizer_mes is not None:
                        self.COWs[idx_cow].optimizer_mes.zero_grad()
                    if self.COWs[idx_cow].optimizer_RT is not None:
                        self.COWs[idx_cow].optimizer_RT.zero_grad()

                    if it < 2:
                        if self.COWs[idx_cow].optimizer_RT is None:
                            self.COWs[idx_cow].create_optimizer(self.lr, "RT")

                        self.losses[idx_cow] += (
                            lambda_mes * self.COWs[idx_cow].compute_mesurement_loss()
                        )
                        loss_mass, loss_pressure = self.COWs[
                            idx_cow
                        ].compute_bifurcation_loss()
                        self.losses[idx_cow] += loss_mass + loss_pressure
                        loss_a0 = self.COWs[idx_cow].compute_a0_loss()
                        self.losses[idx_cow] += loss_a0
                        if idx_cow != max(idx):
                            self.losses[idx_cow].backward(
                                retain_graph=True, inputs=self.COWs[idx_cow].p_RT
                            )
                        else:
                            self.losses[idx_cow].backward(
                                inputs=self.COWs[idx_cow].p_RT
                            )

                        self.COWs[idx_cow].optimizer_RT.step()

                    elif it < int(max_iters / 3) and it >= 2:
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
                        if idx_cow != max(idx):
                            self.losses[idx_cow].backward(
                                retain_graph=True, inputs=self.COWs[idx_cow].p_MES
                            )
                        else:
                            self.losses[idx_cow].backward(
                                inputs=self.COWs[idx_cow].p_MES
                            )

                        self.COWs[idx_cow].optimizer_mes.step()

                    elif it <= int(2 * max_iters / 3) and it >= int(max_iters / 3):

                        if self.COWs[idx_cow].optimizer_non_mes is None:
                            self.COWs[idx_cow].create_optimizer(self.lr, "NON_MES")
                        loss_mass, loss_pressure = self.COWs[
                            idx_cow
                        ].compute_bifurcation_loss()

                        self.losses[idx_cow] += lambda_mass * loss_mass
                        self.losses[idx_cow] += lambda_pressure * loss_pressure
                        loss_a0 = self.COWs[idx_cow].compute_a0_loss()

                        self.losses[idx_cow] += lambda_a0 * loss_a0
                        if idx_cow != max(idx):
                            self.losses[idx_cow].backward(
                                retain_graph=True, inputs=self.COWs[idx_cow].p_NON_MES
                            )
                        else:
                            self.losses[idx_cow].backward(
                                inputs=self.COWs[idx_cow].p_NON_MES
                            )

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

                        self.losses[idx_cow] += lambda_mass * loss_mass
                        self.losses[idx_cow] += lambda_pressure * loss_pressure
                        loss_a0 = self.COWs[idx_cow].compute_a0_loss()

                        self.losses[idx_cow] += lambda_a0 * loss_a0

                        try:
                            if idx_cow != max(idx):
                                self.losses[idx_cow].backward(
                                    retain_graph=True, inputs=self.COWs[idx_cow].p_FULL
                                )
                            else:
                                self.losses[idx_cow].backward(
                                    inputs=self.COWs[idx_cow].p_FULL
                                )

                            self.COWs[idx_cow].optimizer_full.step()

                        except:

                            pass

                    self.COWs[idx_cow].propagate_RT()
                    self.COWs[idx_cow].update_CT()
                    self.COWs[idx_cow].propagate_CT()
                    torch.cuda.empty_cache()
                    gc.collect()

            it += 1

        for idx_cow in range(len(self.COWs)):
            self.validation.append(self.COWs[idx_cow].compute_validation_l2_loss(18))
            self.losses[idx_cow] = self.losses[idx_cow].item()

        for idx_cow in range(len(self.COWs)):
            print(f"Validation loss for cow {idx_cow} = {self.validation[idx_cow]}")
            print(f"Loss for cow {idx_cow} = {self.losses[idx_cow]}")

        self.best_idx = np.argmin(self.losses)

    def dump_solutions(self, path: str):
        """
        Dumps the solutions of the best Circle of Willis (COW) model to the specified path.

        This method solves the arteries for the best COW model (determined by `best_idx`),
        and then dumps the solutions to the given path.

        Args:
            path (str): The file path where the solutions will be saved.

        Returns:
            None
        """
        out, idx = self.COWs[self.best_idx].solve_arteries(18)
        path = str(path.resolve())
        self.COWs[self.best_idx].dump_solutions(path, out, idx=idx)

    def get_L2(self):
        """
        Retrieve the L2 validation value for the best index.

        Returns:
            float: The L2 validation value corresponding to the best index.
        """
        return self.validation[self.best_idx]

    def dump_plots(self, path: str, best: bool = True):
        """
        Dumps the plots of the COWs to the specified path.

        Parameters:
        path (str): The directory path where the plots will be saved.
        best (bool): If True, only the best COW's plots will be dumped.
                     If False, all COWs' plots will be dumped in separate subdirectories.

        Returns:
        None
        """
        if best:
            self.COWs[self.best_idx].dump_plots(path)
        else:
            for idx, cow in enumerate(self.COWs):
                cow.dump_plots(path + f"/COW_{idx}")

    def dump_params(self, path: str, best: bool = True):
        """
        Dumps the parameters of the COWs to the specified path.

        Parameters:
        path (str): The directory path where the parameters will be saved.
        best (bool): If True, dumps the parameters of the best COW. If False, dumps the parameters of all COWs.

        Returns:
        None
        """
        if best:
            self.COWs[self.best_idx].dump_params(path)
        else:
            for idx, cow in enumerate(self.COWs):
                cow.dump_params(path + f"/COW_{idx}")

    def dump_statistics(self, path: str, best: bool = True):
        """
        Dumps the statistics of the COWs to the specified path.

        Parameters:
        path (str): The directory path where the statistics will be saved.
        best (bool): If True, dumps the statistics of the best COW. If False, dumps the statistics of all COWs.
        """
        if best:
            self.COWs[self.best_idx].dump_statistics(path)
        else:
            for idx, cow in enumerate(self.COWs):
                cow.dump_statistics(path + f"/COW_{idx}")

    def dump_validation(self, path: str, arteries, best: bool = True):
        """
        Dumps the validation data of the Circle of Willis (COW) models to the specified path.

        Parameters:
        path (str): The directory path where the validation data will be saved.
        arteries: The arteries data to be validated.
        best (bool): If True, dumps the validation data of the best COW model.
                     If False, dumps the validation data of all COW models. Default is True.

        Returns:
        None
        """
        if best:
            self.COWs[self.best_idx].dump_validation(path, arteries)
        else:
            for idx, cow in enumerate(self.COWs):
                cow.dump_validation(path + f"/COW_{idx}")

    def dump_reconstructed_u_bc_plots(self, path: str, best: bool = True):
        """
        Dumps the reconstructed u_bc plots to the specified path.

        Parameters:
        path (str): The directory path where the plots will be saved.
        best (bool): If True, dumps the plots for the best COW. If False, dumps the plots for all COWs. Default is True.

        Returns:
        None
        """
        if best:
            self.COWs[self.best_idx].dump_reconstructed_u_bc_plots(path)
        else:
            for idx, cow in enumerate(self.COWs):
                cow.dump_reconstructed_u_bc_plots(path + f"/COW_{idx}")

    def get_validation(self, arteries_log: dict, best: bool = True):
        """
        Validate the Circle of Willis (COW) solutions updates the provided arteries log.

        Parameters:
        arteries_log (dict): A dictionary containing the log of arteries data.
        best (bool): If True, validate using the best COW model. Default is True.

        Returns:
        dict: The validation results from the best COW model.

        Raises:
        NotImplementedError: If best is set to False, as validation for all COW models is not implemented.
        """
        if best:
            return self.COWs[self.best_idx].get_validation(arteries_log)
        else:
            raise NotImplementedError
            for idx, cow in enumerate(self.COWs):
                cow.get_validation(arteries_log)
        return arteries_log

    def dump_RT(self, path: str, best: bool = True):
        """
        Dumps the RT (Retention Time) data to the specified path.

        Parameters:
        path (str): The directory path where the RT data will be saved.
        best (bool): If True, dumps the RT data of the best COW (default is True).
                     If False, dumps the RT data of all COWs, each in a separate subdirectory.

        Returns:
        None
        """
        if best:
            self.COWs[self.best_idx].dump_RT(path)
        else:
            for idx, cow in enumerate(self.COWs):
                cow.dump_RT(path + f"/COW_{idx}")

    def dump_mesurement_plots(self, path):
        """
        Dumps measurement plots to the specified path.

        This method calls the `dump_mesurement_plots` method of the best COW
        (determined by `self.best_idx`) and saves the plots to the given path.

        Parameters:
        path (str): The file path where the measurement plots will be saved.

        Returns:
        None
        """
        self.COWs[self.best_idx].dump_mesurement_plots(path)

    def get_best(self):
        """
        Retrieve the best COW (Continuous Wavelet Transform) instance.

        This method returns the COW instance that has been identified as the best
        based on some criteria, which is indicated by the `best_idx` attribute.

        Returns:
            COW: The best COW instance.
        """
        return self.COWs[self.best_idx]
