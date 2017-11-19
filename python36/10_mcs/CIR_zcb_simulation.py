#
# Valuation of Zero-Coupon Bonds by Monte Carlo Simulation
# in Cox-Ingersoll-Ross (1985) Model
# 10_mcs/CIR_zcb_simulation.py
#
# (c) Dr. Yves J. Hilpisch
# Derivatives Analytics with Python
#
import math
import numpy as np
from CIR_zcb_valuation_gen import B
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['font.family'] = 'serif'

#
# Simulation of Square Root Diffusion
#


def CIR_generate_paths(x0, kappa, theta, sigma, T, M, I, x_disc='exact'):
    ''' Function to simulate Square-Root Difussion (SRD/CIR) process.

    Parameters
    ==========
    x0: float
        initial value
    kappa: float
        mean-reversion factor
    theta: float
        long-run mean
    sigma: float
        volatility factor
    T: float
        final date/time horizon
    M: int
        number of time steps
    I: int
        number of paths

    Returns
    =======
    x: NumPy array
        simulated paths
    '''
    dt = T / M
    x = np.zeros((M + 1, I), dtype=np.float)
    x[0] = x0
    xh = np.zeros_like(x)
    xh[0] = x0
    ran = np.random.standard_normal((M + 1, I))

    if x_disc is 'exact':
        # exact discretization
        d = 4 * kappa * theta / sigma ** 2
        c = (sigma ** 2 * (1 - math.exp(-kappa * dt))) / (4 * kappa)
        if d > 1:
            for t in range(1, M + 1):
                l = x[t - 1] * math.exp(-kappa * dt) / c
                chi = np.ramdom.chisquare(d - 1, I)
                x[t] = c * ((ran[t] + np.sqrt(l)) ** 2 + chi)
        else:
            for t in range(1, M + 1):
                l = x[t - 1] * math.exp(-kappa * dt) / c
                N = np.random.poisson(l / 2, I)
                chi = np.random.chisquare(d + 2 * N, I)
                x[t] = c * chi

    else:
        # Euler scheme (full truncation)
        for t in range(1, M + 1):
            xh[t] = (xh[t - 1] + kappa * (theta - np.maximum(0, xh[t - 1])) *
                     dt + np.sqrt(np.maximum(0, xh[t - 1])) *
                     sigma * ran[t] * math.sqrt(dt))
            x[t] = np.maximum(0, xh[t])
    return x


#
# Graphical Output of Simulated Paths
#
def plot_paths():
    plt.figure(figsize=(9, 5))
    plt.plot(range(len(r)), r[:, :20])
    plt.xlabel('time step')
    plt.ylabel('short rate')

#
# Valuation of ZCB
#


def zcb_estimator(M=50, x_disc='exact'):
    dt = T / M
    r = CIR_generate_paths(r0, kappa_r, theta_r, sigma_r, T, M, I, x_disc)
    zcb = np.zeros((M + 1, I), dtype=np.float)
    zcb[-1] = 1.0  # final value
    for t in range(M, 0, -1):
        zcb[t - 1] = zcb[t] * np.exp(-(r[t] + r[t - 1]) / 2 * dt)
    return np.sum(zcb, axis=1) / I

#
# Graphical Value Comparison
#


def graphical_comparison(M=50, x_disc='exact'):
    MCS_values = zcb_estimator(M, x_disc)
    CIR_values = []
    dt = T / M
    t_list = np.arange(0.0, T + 0.001, dt)  # dates of interest
    for t in t_list:
        alpha = r0, kappa_r, theta_r, sigma_r, t, T
        CIR_values.append(B(alpha))
        # CIR model values given date list

    fig, ax = plt.subplots(2, sharex=True, figsize=(8, 6))
    ax[0].plot(t_list, MCS_values, 'ro', label='MCS values')
    ax[0].plot(t_list, CIR_values, 'b', label='CIR values')
    ax[0].legend(loc=0)
    ax[0].set_ylim(min(CIR_values) - 0.005, max(CIR_values) + 0.005)
    ax[0].set_ylabel('option values')
    ax[0].set_title('maturity $T=2$')
    ax[1].bar(t_list - 0.025 / 2., MCS_values - CIR_values,
              width=0.025)
    plt.ylabel('difference')
    plt.xlim(min(t_list) - 0.1, max(t_list) + 0.1)
    plt.xlabel('time $t$')
    plt.tight_layout()


if __name__ == '__main__':
    #
    # Model Parameters
    #
    r0, kappa_r, theta_r, sigma_r = [0.01, 0.1, 0.03, 0.2]
    T = 2.0  # time horizon
    M = 50  # time steps
    dt = T / M
    I = 50000  # number of MCS paths
    np.random.seed(50000)  # seed for RNG

    r = CIR_generate_paths(r0, kappa_r, theta_r, sigma_r, T, M, I)
