import multiprocessing

# from inverse.COW import COW
import sys

import numpy as np
import scipy.integrate

# implementation based on https://github.com/PredictiveIntelligenceLab/1DBloodFlowPINNs


# tu może bcy inny float32 raczej
def get_fft(Q: np.ndarray, T: np.float64):
    """
    Function computes fft coefficients for fourier series approx

    Args:
        Q (np.ndarray): flow (predicted by PINN)
        T (np.float64): period
    """
    n = Q.shape[0] - 1
    yy = Q[0:n]
    N = n
    mN = N // 2
    c = np.fft.fft(yy, N)
    aa = 2 * np.real(c[0 : mN + 1]) / N
    bb = -2 * np.imag(c[0 : mN + 1]) / N
    return aa, bb


def compute_Q_dQ_dt(t, a, b, nmodes, T):
    """
    Function computes fourier series approx of Q
    and its time derivative
    """
    Q = 0.5 * a[0]
    dQ_dt = 0
    for i in range(1, nmodes):
        Q = (
            Q
            + a[i] * np.cos(2 * np.pi * i * t / T)
            + b[i] * np.sin(2 * np.pi * i * t / T)
        )
        dQ_dt = (
            dQ_dt
            - 2 * np.pi * i / T * a[i] * np.sin(2 * np.pi * i * t / T)
            + 2 * np.pi * i / T * b[i] * np.cos(2 * np.pi * i * t / T)
        )
    return Q, dQ_dt


def dydt(p, t, theta, aa, bb, T):
    """
    Function computes dydt for ode solver,
    ode windkessel model, y_prime dp/dt
    """

    Q, dQ_dt = compute_Q_dQ_dt(t, aa, bb, 50, T)

    R1, R2, C, Pinf = theta
    dp_dt = -p / R2 / C + (R1 + R2) / R2 / C * Q + Pinf / R2 / C + R1 * dQ_dt

    return dp_dt


def compute_norm(x, x_pred):
    """
    Function computes L2 norm
    """
    return np.sqrt(np.sum((x - x_pred.squeeze()) ** 2)) / np.sqrt(np.sum(x**2))


def search_params(R, C, t, p_exact, p, nn, R1, p_inf, T, aa, bb):
    """
    Function performs parameter serach

    R, C: search space
    p_exact: exact pressure from tranined PINN (for ode solver) dQ_dt dim than p
    p: pressure from trained PINN
    nn: number of samples
    t: time
    """
    # counter jest w sumie zbędny? wyjebiemy to by było
    # szybciej i bardziej czytelnie
    counter = 0
    error = np.zeros((R.shape[0], C.shape[0]))
    p_best = dict()
    for i in range(R.shape[0]):
        for j in range(C.shape[0]):
            truth = np.array([R1, R[i, j], C[i, j], p_inf])
            p_pred = scipy.integrate.odeint(dydt, p_exact[0], t, (truth, aa, bb, T))
            error[i, j] = compute_norm(p, p_pred)
            p_best[str(i), str(j)] = p_pred
            counter = counter + 1
            print("Sample %d/%d" % (counter, nn**2))
    idx_row, idx_col = np.where(error == np.amin(error[np.nonzero(error)]))
    best_R, best_C = R[idx_row, idx_col], C[idx_row, idx_col]
    p_ = p_best[str(idx_row.item()), str(idx_col.item())]
    return best_R, best_C, error, p_


def create_search_grid(R_upper_bound, R_lower_bound, C_upper_bound, C_lower_bound, nn):
    """
    Function creates search grid for parameter search

    Args:
        R_upper_bound (float): upper bound for R
        R_lower_bound (float): lower bound for R
        C_upper_bound (float): upper bound for C
        C_lower_bound (float): lower bound for C
        nn (int): number of samples
    """
    R_prop = np.exp(np.linspace(R_lower_bound, R_upper_bound, nn))
    C_prop = np.exp(np.linspace(C_lower_bound, C_upper_bound, nn))
    R, C = np.meshgrid(R_prop, C_prop)
    return R, C


# truth = [R1, R2, C, Pinf]
# trzeba do ode solvera jakos dac aa, bb, T,
# # scipy.integrate.odeint(dydt, p_exact[0], x, (truth,aa, bb, T))


def parallel_search_params(args):
    i, j, R1, t, p_exact, p, T, aa, bb, R, C, p_inf = args
    truth = np.array([R1, R[i, j], C[i, j], p_inf])
    p_pred = scipy.integrate.odeint(dydt, p_exact[0], t, (truth, aa, bb, T))
    error = compute_norm(p, p_pred)
    return i, j, error, p_pred


def search_params_parallel(R, C, t, p_exact, p, nn, R1, p_inf, T, aa, bb):
    pool = multiprocessing.Pool(
        processes=multiprocessing.cpu_count()
    )  # Number of CPU cores
    args_list = [
        (i, j, R1, t, p_exact, p, T, aa, bb, R, C, p_inf)
        for i in range(R.shape[0])
        for j in range(C.shape[0])
    ]
    results = pool.map(parallel_search_params, args_list)
    pool.close()
    pool.join()

    error = np.zeros((R.shape[0], C.shape[0]))
    p_best = dict()
    for i, j, err, p_pred in results:
        error[i, j] = err
        p_best[str(i), str(j)] = p_pred

    idx_row, idx_col = np.where(error == np.amin(error[np.nonzero(error)]))
    best_R, best_C = R[idx_row, idx_col], C[idx_row, idx_col]
    p_ = p_best[str(idx_row.item()), str(idx_col.item())]
    return best_R, best_C, error, p_


def find_windkessel(cow, num_iters: int):
    """
    Function performs adaptive parameter search for windkessel model

    Args:
        an (Artery_network): Artery network object
        num_iters (int): number of iterations for adaptive search

    Returns:
        R2 (np.array): array of R2 values for each artery
        C (np.array): array of C values for each artery
        Z (np.array): array of Z values for each artery
    """
    R2, C, Z = (
        np.zeros(cow.get_num_arteries()),
        np.zeros(cow.get_num_arteries()),
        np.zeros(cow.get_num_arteries()),
    )
    p_inf = 666.5
    rho = 1050.0
    out_pred_dict = cow.get_outlet_predictions(True)

    for key, value in out_pred_dict.items():
        u, A, p, t = value

        u, A, p, t = (
            u.squeeze(),
            A.squeeze(),
           
            p.squeeze(),
            t.squeeze(),
        )

        Q = np.multiply(A, u)
        T = np.max(t)
        p_exact = p
        p = p_exact[:]
        # print(Q.shape)
        NN = Q.shape[0]
        z = compute_characteristic_impedance(cow.get_artery(int(key)), rho)
        print(key)
        print(z)
        print(T)

        # compute fft
        aa, bb = get_fft(Q, T)

        t = t.flatten()

        # initial coarse search
        # TODO na podstawie danych literaturowych oszacować zakres wyszukiwania
        # dodanmy gdy tim kliniczny zrobi robote
        import time

        nn = 10
        lb = np.array([19.0, -19.0])
        ub = np.array([29.0, -29.0])
        R_prop = np.exp(np.linspace(lb[0], ub[0], nn))
        C_prop = np.exp(np.linspace(lb[1], ub[1], nn))
        R_temp, C_temp = np.meshgrid(R_prop, C_prop)
        # st = time.time()
        best_R, best_C, error, p_ = search_params_parallel(
            R_temp, C_temp, t, p_exact, p, nn, z, p_inf, T, aa, bb
        )

        # best_R, best_C, error, p_ = search_params(
        #    R_temp, C_temp, t, p_exact, p, nn, z, p_inf, T, aa, bb
        # )

        # adaptively refined search
        for i in range(0, num_iters):
            lb = np.array([best_R - 0.5 * best_R, best_C - 0.5 * best_C])
            ub = np.array([best_R + 0.5 * best_R, best_C + 0.5 * best_C])
            R_prop = np.linspace(lb[0], ub[0], nn)
            C_prop = np.linspace(lb[1], ub[1], nn)
            R_temp, C_temp = np.meshgrid(R_prop, C_prop)
            best_R, best_C, error, p_ = search_params_parallel(
                R_temp, C_temp, t, p_exact, p, nn, z, p_inf, T, aa, bb
            )
            # best_R, best_C , error , p_ = search_params(R_temp, C_temp, t, p_exact, p, nn, z, p_inf, T, aa, bb)

        idx_row, idx_col = np.where(error == np.amin(error[np.nonzero(error)]))
        print(f"best error {error[idx_row, idx_col]}")
        R2[int(key)] = best_R
        C[int(key)] = best_C
        Z[int(key)] = z

    return R2, C, Z


def compute_beta(r0: float):
    """
    Function computes Eh from empirical relation Olufsen
    """
    k1 = 0.3e6
    k2 = -1350
    k3 = 43.7e3
    Eh = r0 * (k1 * np.exp(k2 * r0) + k3)
    return 4 / 3 * np.sqrt(np.pi) * Eh


def compute_c0(r0: float, beta: float, rho: float):
    """
    Function computes  pulswave propagation speed at eq
    """
    return np.sqrt(beta / (2 * rho * np.sqrt(np.pi * r0**2)))


def compute_characteristic_impedance(Artery, rho: float):
    """
    Function computes characteristic impedance for artery
    """
    beta = compute_beta((Artery.get_r0() * 1e-2))
    c0 = compute_c0((Artery.get_r0() * 1e-2), beta, rho)
    return rho * c0 / (np.pi * (Artery.get_r0() * 1e-2) ** 2)


##### UWAGA NA JEDNOSTKI DO SOLVERA !!!! CHYBA SA SI !!!!
