#
# Valuation of European Options
# Under Heston (1993) Stochastic Volatility Model
# Comparison of Fourier-based Value and MCS Estimator
# 10_mcs/H93_european_mcs.py
#
# (c) Dr. Yves J. Hilpisch
# Derivatives Analytics with Python
#
import sys
sys.path.append('09_gmm/')
import math
import string
import numpy as np
np.set_printoptions(precision=3)
import pandas as pd
import itertools as it
from datetime import datetime
from time import time
from BCC_option_valuation import H93_call_value

# Fixed Short Rate
r = 0.05

# Heston (1993) Parameters
# from MS (2009), table 3
para = np.array(((0.01, 1.5, 0.15, 0.1),  # panel 1
             # (v0,kappa_v,sigma_v,rho)
              (0.04, 0.75, 0.3, 0.1),  # panel 2
              (0.04, 1.50, 0.3, 0.1),  # panel 3
              (0.04, 1.5, 0.15, -0.5)))  # panel 4

theta_v = 0.02  # long-term variance level
S0 = 100.0  # initial index level

# General Simulation Parameters
write = True
verbose = False
option_types = ['CALL', 'PUT']  # option types
steps_list = [25, 50]  # time steps p.a.
paths_list = [25000, 50000, 75000, 100000]   # number of paths per valuation

s_disc_list = ['Log', 'Naive']  # Euler scheme: log vs. naive
x_disc_list = ['Full Truncation', 'Partial Truncation', 'Truncation',
               'Absorption', 'Reflection', 'Higham-Mao', 'Simple Reflection']
                # discretization schemes for SRD process

anti_paths = [False, True] 
  # antithetic paths for variance reduction
moment_matching = [False, True] 
  # random number correction (std + mean + drift)

t_list = [1.0 / 12, 1.0, 2.0]  # maturity list
k_list = [90, 100, 110]      # strike list
PY1 = 0.025  # performance yardstick 1: abs. error in currency units
PY2 = 0.015  # performance yardstick 2: rel. error in decimals
runs = 5  # number of simulation runs
np.random.seed(250000)  # set RNG seed value

#
# Function for Short Rate and Volatility Processes
#


def SRD_generate_paths(x_disc, x0, kappa, theta, sigma,
                       T, M, I, rand, row, cho_matrix):
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
    for t in xrange(1, M + 1):
        ran = np.dot(cho_matrix, rand[:, t])
        if x_disc == 'Full Truncation':
            xh[t] = (xh[t - 1] + kappa * (theta -
                 np.maximum(0, xh[t - 1])) * dt +
                 np.sqrt(np.maximum(0, xh[t - 1])) * sigma * ran[row] * sdt)
            x[t] = np.maximum(0, xh[t])
        elif x_disc == 'Partial Truncation':
            xh[t] = (xh[t - 1] + kappa * (theta - xh[t - 1]) * dt +
                 np.sqrt(np.maximum(0, xh[t - 1])) * sigma * ran[row] * sdt)
            x[t] = np.maximum(0, xh[t])
        elif x_disc == 'Truncation':
            x[t] = np.maximum(0, x[t - 1]
                 + kappa * (theta - x[t - 1]) * dt +
                 np.sqrt(x[t - 1]) * sigma * ran[row] * sdt)
        elif x_disc == 'Reflection':
            xh[t] = (xh[t - 1]
                 + kappa * (theta - abs(xh[t - 1])) * dt +
                 np.sqrt(abs(xh[t - 1])) * sigma * ran[row] * sdt)
            x[t] = abs(xh[t])
        elif x_disc == 'Higham-Mao':
            xh[t] = (xh[t - 1] + kappa * (theta - xh[t - 1]) * dt +
                 np.sqrt(abs(xh[t - 1])) * sigma * ran[row] * sdt)
            x[t] = abs(xh[t])
        elif x_disc == 'Simple Reflection':
            x[t] = abs(x[t - 1] + kappa * (theta - x[t - 1]) * dt +
                 np.sqrt(x[t - 1]) * sigma * ran[row] * sdt)
        elif x_disc == 'Absorption':
            xh[t] = (np.maximum(0, xh[t - 1])
                 + kappa * (theta - np.maximum(0, xh[t - 1])) * dt +
                 np.sqrt(np.maximum(0, xh[t - 1])) * sigma * ran[row] * sdt)
            x[t] = np.maximum(0, xh[t])
        else:
            print x_disc
            print "No valid Euler scheme."
            sys.exit(0)
    return x

#
# Function for Heston Index Process
#


def H93_generate_paths(S0, r, v, row, cho_matrix):
    ''' Simulation of Heston (1993) index process.
    
    Parameters
    ==========
    S0: float
        initial value
    r: float
        constant short rate
    v: NumPy array
        simulated variance paths
    row: int
        row/matrix of random number array to use
    cho_matrix: NumPy array
        Cholesky matrix

    Returns
    =======
    S: NumPy array
        simulated index level paths
    '''
    S = np.zeros((M + 1, I), dtype=np.float)
    S[0] = S0
    bias = 0.0
    sdt = math.sqrt(dt)
    for t in xrange(1, M + 1, 1):
        ran = np.dot(cho_matrix, rand[:, t])
        if momatch:
            bias = np.mean(np.sqrt(v[t]) * ran[row] * sdt)
        if s_disc == 'Log':
            S[t] = S[t - 1] * np.exp((r - 0.5 * v[t]) * dt +
                    np.sqrt(v[t]) * ran[row] * sdt - bias)
        elif s_disc == 'Naive':
            S[t] = S[t - 1] * (math.exp(r * dt) +
                    np.sqrt(v[t]) * ran[row] * sdt - bias)
        else:
            print "No valid Euler scheme."
            exit(0)
    return S


def random_number_generator(M, I):
    ''' Function to generate pseudo-random numbers.

    Parameters
    ==========
    M: int
        time steps
    I: int
        number of simulation paths

    Returns
    =======
    rand: NumPy array
        random number array
    '''
    if antipath:
        rand = np.random.standard_normal((2, M + 1, I / 2))
        rand = np.concatenate((rand, -rand), 2)
    else:
        rand = np.random.standard_normal((2, M + 1, I))
    if momatch:
        rand = rand / np.std(rand)
        rand = rand - np.mean(rand)
    return rand

#
# Valuation
#

t0 = time()

results = pd.DataFrame()

tmpl_1 = '%4s | %3s | %6s | %6s | %6s | %6s | %5s | %5s' \
            % ('T', 'K', 'V0', 'V0_MCS', 'err', 'rerr', 'acc1', 'acc2')
tmpl_2 = '%4.3f | %3d | %6.3f | %6.3f | %6.3f | %6.3f | %5s | %5s'

if __name__ == '__main__':

    for alpha in it.product(option_types, steps_list, paths_list, s_disc_list,
                            x_disc_list, anti_paths, moment_matching):
        print '\n\n', alpha, '\n'
        option, M0, I, s_disc, x_disc, antipath, momatch = alpha
        for run in range(runs):
            for panel in range(4):
                # Correlation Matrix
                v0, kappa_v, sigma_v, rho = para[panel]
                covariance_matrix = np.zeros((2, 2), dtype=np.float)
                covariance_matrix[0] = [1.0, rho]
                covariance_matrix[1] = [rho, 1.0]
                cho_matrix = np.linalg.cholesky(covariance_matrix)
                if verbose:
                    print "\nResults for Panel %d\n" % (panel + 1)
                    print tmpl_1
                for T in t_list:  # maturity list
                    # memory clean-up
                    v, S, rand, h = 0.0, 0.0, 0.0, 0.0
                    M = int(M0 * T)  # number of total time steps 
                    dt = T / M  # time interval in years
                    # random numbers
                    rand = random_number_generator(M, I)
                    # volatility process paths
                    v = SRD_generate_paths(x_disc, v0, kappa_v, theta_v,
                                    sigma_v, T, M, I, rand, 1, cho_matrix)
                    # index level process paths
                    S = H93_generate_paths(S0, r, v, 0, cho_matrix)
                    for K in k_list:
                        # European option values
                        B0T = math.exp(-r * T)  # discount factor
                        # European call option value (semi-analytical)
                        C0 = H93_call_value(S0, K, T, r, kappa_v,
                                             theta_v, sigma_v, rho, v0)
                        P0 = C0 + K * B0T - S0
                        if option is 'PUT':
                            # benchmark value
                            V0 = P0
                            # inner value matrix put
                            h = np.maximum(K - S, 0)
                        elif option is 'CALL':
                            # benchmark value
                            V0 = C0
                            # inner value matrix call
                            h = np.maximum(S - K, 0)
                        else:
                            print "No valid option type."
                            sys.exit(0)
                        pv = B0T * h[-1]  # present value vector
                        V0_MCS = np.sum(pv) / I   # MCS estimator
                        SE = np.std(pv) / math.sqrt(I)
                          # standard error
                        error = V0_MCS - V0
                        rel_error = (V0_MCS - V0) / V0
                        PY1_acc = abs(error) < PY1
                        PY2_acc = abs(rel_error) < PY2
                        res = pd.DataFrame({'timestamp': datetime.now(),
                            'otype': option, 'runs': runs, 'steps': M0,
                            'paths': I, 'index_disc': s_disc,
                            'var_disc': x_disc, 'anti_paths': antipath,
                            'moment_matching': momatch, 'panel': panel,
                            'maturity': T, 'strike': K, 'value': V0,
                            'MCS_est': V0_MCS, 'SE': SE, 'error': error,
                            'rel_error': rel_error, 'PY1': PY1, 'PY2': PY2,
                            'PY1_acc': PY1_acc, 'PY2_acc': PY2_acc,
                            'PY_acc': PY1_acc or PY2_acc},
                            index=[0,])
                        
                        if verbose:
                            print tmpl_2 % (T, K, V0, V0_MCS, error, 
                                        rel_error, PY1_acc, PY2_acc)
                        
                        results = results.append(res, ignore_index=True)

    if write:
        d = str(datetime.now().replace(microsecond=0))
        d = d.translate(string.maketrans("-:", "__"))
        h5 = pd.HDFStore('10_mcs/mcs_european_%s_%s.h5'
                            % (d[:10], d[11:]), 'w')
        h5['results'] = results
        h5.close()

    print "Total time in minutes %8.2f" % ((time() - t0) / 60)
