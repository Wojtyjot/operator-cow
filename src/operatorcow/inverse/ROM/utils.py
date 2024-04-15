import os
import sys
from typing import *

import numpy as np
import pandas as pd
from ruamel.yaml.main import round_trip_dump as yaml_dump

# Przydałoby sie zunifikowac joints w artery network i w openBF


def create_simulation_script(df: pd.DataFrame, project_name: str):
    """
    Func creates yaml file for openBF from pandas dataframe
    """
    # TODO zmienic parametry windkessela
    # te parametry trzeba dac do jakiegos pliku i czytac
    # chyba ze zostawimy hardcoded i wyjebane
    # dodac p_ext do pliku
    P_ext = None

    # hardcoded solver params
    rho = 1050.0
    mu = 4.5e-3
    Ccfl = 0.9  # Courant number
    cycles = 100  # number of cycles
    jump = 100  # number of timesteps per cycle to be saved
    convergence_tolerance = 1.0  # percentage value
    network_elements = list()
    for i in df.id:
        if (df[df.id == i].inlet != 0).bool():
            network_elements.append(
                {
                    "label": df[df.id == i].name.iloc[0],
                    "sn": int(df[df.id == i].sn.iloc[0]),
                    "tn": int(df[df.id == i].tn.iloc[0]),
                    "L": float(df[df.id == i].L.iloc[0]),
                    "E": float(df[df.id == i].E.iloc[0]),
                    "Rp": float(df[df.id == i].Rp.iloc[0]),
                    "Rd": float(df[df.id == i].Rd.iloc[0]),
                    "M": int(df[df.id == i].M.iloc[0]),
                    "inlet": "Q",
                    "inlet number": int(df[df.id == i].inlet.iloc[0]),
                    "inlet file": df[df.id == i].inlet_file.iloc[0],
                }
            )
        elif (df[df.id == i].outlet == 1).bool():
            network_elements.append(
                {
                    "label": df[df.id == i].name.iloc[0],
                    "sn": int(df[df.id == i].sn.iloc[0]),
                    "tn": int(df[df.id == i].tn.iloc[0]),
                    "L": float(df[df.id == i].L.iloc[0]),
                    "E": float(df[df.id == i].E.iloc[0]),
                    "Rp": float(df[df.id == i].Rp.iloc[0]),
                    "Rd": float(df[df.id == i].Rd.iloc[0]),
                    "M": int(df[df.id == i].M.iloc[0]),
                    "outlet": "wk3",
                    "R1": float(df[df.id == i].R1.iloc[0]),
                    "R2": float(df[df.id == i].R2.iloc[0]),
                    "Cc": float(df[df.id == i].C.iloc[0]),
                }
            )
        else:
            network_elements.append(
                {
                    "label": df[df.id == i].name.iloc[0],
                    "sn": int(df[df.id == i].sn.iloc[0]),
                    "tn": int(df[df.id == i].tn.iloc[0]),
                    "L": float(df[df.id == i].L.iloc[0]),
                    "E": float(df[df.id == i].E.iloc[0]),
                    "Rp": float(df[df.id == i].Rp.iloc[0]),
                    "Rd": float(df[df.id == i].Rd.iloc[0]),
                    "M": int(df[df.id == i].M.iloc[0]),
                }
            )
    data = {
        "project name": project_name,
        "blood": {
            "rho": rho,
            "mu": mu,
        },
        "solver": {
            "Ccfl": Ccfl,
            "cycles": cycles,
            "jump": jump,
            "convergence tolerance": convergence_tolerance,
        },
        "network": network_elements,
    }
    with open(f"{project_name}.yaml", "w") as yaml_file:
        yaml_dump(data, yaml_file, default_flow_style=False)


def create_Julia_file(project_name: str, src: str):
    """
    Function creates .jl file for running openBF simulation

    src: path to modified openBF package src folder
    """
    with open(f"{project_name}.jl", "w") as f:
        f.write(f'include("{src}/openBF.jl")\n')
        f.write(f'conf_file = "{project_name}.yaml"\n')
        f.write(f"openBF.runSimulation(conf_file, verbose=true)\n")


def run_simulation(project_name: str):
    """
    Functon runs openBF simulation
    """
    os.system(f"julia {project_name}.jl")


def compute_R_tot(P_sys: float, P_dia: float, SV: float, HR: float):
    """
    Function computes total resistance of arterial network
    """
    return (1 / 3 * P_sys + 2 / 3 * P_dia) / (SV * HR)


def compute_C_tot(R_tot: float, tau: Optional[float] = 1.34):
    """
    Function computes total compliance

    tau: [s] aortic pressure decay
    """
    return tau / R_tot


def compute_c0(r0: float, beta: float, rho: float):
    """
    Function computes  pulswave propagation speed at eq
    """
    return np.sqrt(beta / (2 * rho * np.sqrt(np.pi * r0**2)))


def compute_beta(r0: float):
    """
    Function computes Eh from empirical relation Olufsen
    """
    k1 = 2e6
    k2 = -2253
    k3 = 86.5e3
    Eh = r0 * (k1 * np.exp(k2 * r0) + k3)
    return 4 / 3 * np.sqrt(np.pi) * Eh


def compute_R1(rho: float, c0: float, r0: float):
    """
    Function computes R1
    """
    return rho * c0 / (np.pi * r0**2)


def compute_R_t(R_tot: float, COf: float, sigma_r0: float, r0: float):
    """
    Func computes R_t based on
    R. M. Padmos, T. I. Józsa, W. K. El-Bouri, P. R. Konduri, S. J. Payne,
    and A. G. Hoekstra, “Coupling one-dimensional arterial blood flow to
    three-dimensional tissue perfusion models for in silico trials of acute
    ischaemic stroke,” Interface focus, vol. 11, no. 1, p. 20190125, 2021
    """
    return R_tot / COf * sigma_r0 / r0


def compute_C_t(C_tot: float, R_tot: float, R_t: float):
    """
    Func computes arterial compliance based on
    R. M. Padmos, T. I. Józsa, W. K. El-Bouri, P. R. Konduri, S. J. Payne,
    and A. G. Hoekstra, “Coupling one-dimensional arterial blood flow to
    three-dimensional tissue perfusion models for in silico trials of acute
    i   schaemic stroke,” Interface focus, vol. 11, no. 1, p. 20190125, 2021
    """
    return C_tot * R_tot / R_t


def compute_R_2(R_t: float, R_1: float):
    """
    Func computes R2 for windkessel model
    """
    return R_t - R_1
