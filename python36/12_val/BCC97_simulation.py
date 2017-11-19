#
# Monte Carlo Simulation of BCC97 Model
# 12_val/BCC97_simulation.py
#
# (c) Dr. Yves J. Hilpisch
# Derivatives Analytics with Python
#
import sys
import math
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
sys.path.append('11_cal')
from H93_calibration import S0, kappa_r, theta_r, sigma_r, r0

mpl.rcParams['font.family'] = 'serif'
#
# Model Parameters
#
opt = np.load('11_cal/opt_full.npy')
kappa_v, theta_v, sigma_v, rho, v0, lamb, mu, delta = opt


#
# Simulation Parameters
#
T = 1.0  # time horizon
M = 25  # time steps
I = 10000  # number of replications per valuation
anti_paths = True  # antithetic paths for variance reduction
moment_matching = True  # moment matching for variance reduction
np.random.seed(100000)  # seed value for random number generator

#
# Random Number Generation
#


def generate_cholesky(rho):
    ''' Function to generate Cholesky matrix.

    Parameters
    ==========
    rho: float
        correlation between index level and variance

    Returns
    =======
    matrix: NumPy array
        Cholesky matrix
    '''
    rho_rs = 0  # correlation between index level and short rate
    covariance = np.zeros((4, 4), dtype=np.float)
    covariance[0] = [1.0, rho_rs, 0.0, 0.0]
    covariance[1] = [rho_rs, 1.0, rho, 0.0]
    covariance[2] = [0.0, rho, 1.0, 0.0]
    covariance[3] = [0.0, 0.0, 0.0, 1.0]
    cho_matrix = np.linalg.cholesky(covariance)
    return cho_matrix


def random_number_generator(M, I, anti_paths, moment_matching):
    ''' Function to generate pseudo-random numbers.

    Parameters
    ==========
    M: int
        time steps
    I: int
        number of simulation paths
    anti_paths: bool
        flag for antithetic paths
    moment_matching: bool
        flag for moment matching

    Returns
    =======
    rand: NumPy array
        random number array
    '''
    if anti_paths:
        rand = np.random.standard_normal((4, M + 1, int(I / 2)))
        rand = np.concatenate((rand, -rand), 2)
    else:
        rand = np.random.standard_normal((4, M + 1, I))
    if moment_matching:
        for a in range(4):
            rand[a] = rand[a] / np.std(rand[a])
            rand[a] = rand[a] - np.mean(rand[a])
    return rand

#
# Function for Short Rate and Volatility Processes
#


def SRD_generate_paths(x0, kappa, theta, sigma, T, M, I,
                       rand, row, cho_matrix):
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
    row: int
        row number for random numbers
    cho_matrix: NumPy array
        cholesky matrix

    Returns
    =======
    x: NumPy array
        simulated variance paths
    '''
    dt = T / M
    x = np.zeros((M + 1, I), dtype=np.float)
    x[0] = x0
    xh = np.zeros_like(x)
    xh[0] = x0
    sdt = math.sqrt(dt)
    for t in range(1, M + 1):
        ran = np.dot(cho_matrix, rand[:, t])
        xh[t] = (xh[t - 1] + kappa * (theta -
                                      np.maximum(0, xh[t - 1])) * dt +
                 np.sqrt(np.maximum(0, xh[t - 1])) * sigma * ran[row] * sdt)
        x[t] = np.maximum(0, xh[t])
    return x

#
# Function for B96 Index Process
#


def B96_generate_paths(S0, r, v, lamb, mu, delta, rand, row1, row2,
                       cho_matrix, T, M, I, moment_matching):
    ''' Simulation of Bates (1996) index process.

    Parameters
    ==========
    S0: float
        initial value
    r: NumPy array
        simulated short rate paths
    v: NumPy array
        simulated variance paths
    lamb: float
        jump intensity
    mu: float
        expected jump size
    delta: float
        standard deviation of jump
    rand: NumPy array
        random number array
    row1, row2: int
        rows/matrices of random number array to use
    cho_matrix: NumPy array
        Cholesky matrix
    T: float
        time horizon, maturity
    M: int
        number of time intervals, steps
    I: int
        number of paths to simulate
    moment_matching: bool
        flag for moment matching

    Returns
    =======
    S: NumPy array
        simulated index level paths
    '''
    S = np.zeros((M + 1, I), dtype=np.float)
    S[0] = S0
    dt = T / M
    sdt = math.sqrt(dt)
    ranp = np.random.poisson(lamb * dt, (M + 1, I))
    bias = 0.0
    for t in range(1, M + 1, 1):
        ran = np.dot(cho_matrix, rand[:, t, :])
        if moment_matching:
            bias = np.mean(np.sqrt(v[t]) * ran[row1] * sdt)
        S[t] = S[t - 1] * (np.exp(((r[t] + r[t - 1]) / 2 - 0.5 * v[t]) * dt +
                                  np.sqrt(v[t]) * ran[row1] * sdt - bias) +
                           (np.exp(mu + delta * ran[row2]) - 1) * ranp[t])
    return S


if __name__ == '__main__':
    #
    # Simulation
    #
    cho_matrix = generate_cholesky(rho)
    rand = random_number_generator(M, I, anti_paths, moment_matching)
    r = SRD_generate_paths(r0, kappa_r, theta_r, sigma_r, T, M, I,
                           rand, 0, cho_matrix)
    v = SRD_generate_paths(v0, kappa_v, theta_v, sigma_v, T, M, I,
                           rand, 2, cho_matrix)
    S = B96_generate_paths(S0, r, v, lamb, mu, delta, rand, 1, 3,
                           cho_matrix, T, M, I, moment_matching)


def plot_rate_paths(r):
    plt.figure(figsize=(10, 6))
    plt.plot(r[:, :10])
    plt.xlabel('time step')
    plt.ylabel('short rate level')
    plt.title('Short Rate Simulated Paths')


def plot_volatility_paths(v):
    plt.figure(figsize=(10, 6))
    plt.plot(np.sqrt(v[:, :10]))
    plt.xlabel('time step')
    plt.ylabel('volatility level')
    plt.title('Volatility Simulated Paths')


def plot_index_paths(S):
    plt.figure(figsize=(10, 6))
    plt.plot(S[:, :10])
    plt.xlabel('time step')
    plt.ylabel('index level')
    plt.title('EURO STOXX 50 Simulated Paths')


def plot_index_histogram(S):
    plt.figure(figsize=(10, 6))
    plt.hist(S[-1], bins=30)
    plt.xlabel('index level')
    plt.ylabel('frequency')
    plt.title('EURO STOXX 50 Values after 1 Year')
