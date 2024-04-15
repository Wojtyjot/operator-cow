from typing import *

import numpy as np
import pandas as pd
from ROM.utils import (
    compute_beta,
    compute_c0,
    compute_C_t,
    compute_C_tot,
    compute_R1,
    compute_R_2,
    compute_R_t,
    compute_R_tot,
    create_Julia_file,
    create_simulation_script,
    run_simulation,
)


def estimate_windkessel_func(
    arteries_csv: str,
    P_sys: float,
    P_dia: float,
    SV: float,
    HR: float,
    rho: float,
    tau: Optional[float] = 1.34,
    COf: Optional[float] = 0.12,
):
    """
    Function estimates windkessel parameters using strategy from paper
    M. Sarabian, H. Babaee and K. Laksari,
    "Physics-Informed Neural Networks for Brain Hemodynamic Predictions Using Medical Imaging,"
    in IEEE Transactions on Medical Imaging,
    vol. 41, no. 9, pp. 2285-2303, Sept. 2022,
    doi: 10.1109/TMI.2022.3161653.
    """
    ### w tej formie nadaje sie tylko do estymacji COW
    ### aby estymowaac w t ramiennych i aorcie trzeba zmienic
    ### plik csv moze miec tylko outlety w glowie
    df = pd.read_csv(arteries_csv)
    R1, R2, C = list(), list(), list()
    r0 = list()
    R_tot = compute_R_tot(P_sys, P_dia, SV, HR)
    C_tot = compute_C_tot(R_tot, tau)

    for id in df.id:
        if (df[df.id == id].outlet == 1).bool():
            r0.append(df[df.id == id].Rd.iloc[0])
    sigma_r0 = np.sum(np.array(r0))
    for id in df.id:
        if (df[df.id == id].outlet == 1).bool():
            r_0 = df[df.id == id].Rd.iloc[0]
            beta = compute_beta(r_0)
            c0 = compute_c0(r_0, beta, rho)
            R_1 = compute_R1(rho, c0, r_0)
            R_T = compute_R_t(R_tot, COf, sigma_r0, r_0)
            R_2 = compute_R_2(R_T, R_1)
            C_T = compute_C_t(C_tot, R_tot, R_T)
            R1.append(R_1)
            R2.append(R_2)
            C.append(C_T)
        else:
            R1.append(0)
            R2.append(0)
            C.append(0)

    df["R1"] = R1
    df["R2"] = R2
    df["C"] = C
    return df


def apply_windkessel(arteries_csv: str, R1: list, R2: list, C: list):
    """
    Function appends specified windkessel parameters to data

    len of R1, R2, C must match len of arteries
    0 if artery is not an outlet
    """
    df = pd.read_csv(arteries_csv)
    # check lengths
    if len(R1) != len(df.id):
        raise ValueError("R1 list length does not match length of arteries")
    elif len(R2) != len(df.id):
        raise ValueError("R2 list length does not match length of arteries")
    elif len(C) != len(df.id):
        raise ValueError("C list length does not match length of arteries")
    else:
        df["R1"] = R1
        df["R2"] = R2
        df["C"] = C
    return df


def Simulation(
    project_name: str,
    arteries_csv: str,
    estimate_windkessel: bool,
    src: str,
    P_sys: Optional[float] = 17331.91,
    P_dia: Optional[float] = 11999.01,
    SV: Optional[float] = 0.00007,
    HR: Optional[float] = 1,
    rho: Optional[float] = 1050,
    tau: Optional[float] = 1.34,
    COf: Optional[float] = 0.12,
    R1: Optional[list] = None,
    R2: Optional[list] = None,
    C: Optional[list] = None,
):
    """
    Function creates simulation script and runs it
    """
    if estimate_windkessel:
        df = estimate_windkessel_func(arteries_csv, P_sys, P_dia, SV, HR, rho, tau, COf)
    else:
        if R1 is None or R2 is None or C is None:
            raise ValueError(
                "R1, R2, C must be specified when estimate_windkessel is False"
            )
        df = apply_windkessel(arteries_csv, R1, R2, C)
    print("Creating simulation script...")
    create_simulation_script(df, project_name)
    print("Script created")
    print("Creating Julia file")
    create_Julia_file(project_name, src)
    print("Julia file created")
    print("Running simulation...")
    run_simulation(project_name)
    print("Simulation finished")
