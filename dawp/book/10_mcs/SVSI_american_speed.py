#
# Script for American Put Option Valuation by MCS/LSM
# in H93 and CIR85 model
#
# Examples from Medvedev & Scaillet (2010):
# "Pricing American Options Under Stochastic Volatility
# and Stochastic Interest Rates."
#
# 10_mcs/SVSI_american_mcs.py
#
# (c) Dr. Yves J. Hilpisch
# Derivatives Analytics with Python
#
import gc
import sys
sys.path.append('09_gmm/')
import math
import string
import numpy as np
import pandas as pd
import itertools as it
from datetime import datetime
from BCC_option_valuation import H93_call_value
from H93_european_mcs import SRD_generate_paths
from CIR_zcb_valuation_gen import B
from time import time

# 'True' American Options Prices by Monte Carlo
# from MS (2009), table 3
benchmarks = np.array(((0.0001, 1.0438, 9.9950, 0.0346, 1.7379, 9.9823,
                 0.2040, 2.3951, 9.9726),       # panel 1
                (0.0619, 2.1306, 10.0386, 0.5303, 3.4173, 10.4271,
                 1.1824, 4.4249, 11.0224),      # panel 2
                (0.0592, 2.1138, 10.0372, 0.4950, 3.3478, 10.3825,
                 1.0752, 4.2732, 10.8964),      # panel 3
                (0.0787, 2.1277, 10.0198, 0.6012, 3.4089, 10.2512,
                 1.2896, 4.4103, 10.6988)))    # panel 4

# Cox, Ingersoll, Ross (1985) Parameters
# from MS (2009), table 3, panel 1
r0 = 0.04
kappa_r = 0.3
theta_r = 0.04
sigma_r = 0.1

# Heston (1993) Parameters
# from MS (2009), table 3
para = np.array(((0.01, 1.50, 0.15, 0.10),  # panel 1
            # (v0, kappa_v, sigma_v, rho)
            (0.04, 0.75, 0.30, 0.10),  # panel 2
            (0.04, 1.50, 0.30, 0.10),   # panel 3
            (0.04, 1.50, 0.15, -0.50)))  # panel 4

theta_v = 0.02  # long-term variance level
S0 = 100.0  # initial index level
D = 10  # number of basis functions

# General Simulation Parameters
write = False
verbose = False

py_list = [(0.025, 0.015)] # , (0.01, 0.01)]
  # combinations of performance yardsticks (absolute, relative)
  # performance yardstick 1: abs. error in currency units
  # performance yardstick 2: rel. error in decimals

m_list = [25]  # number of time intervals
paths_list = [35000]  # number of paths per valuation

x_disc_list = ['Full Truncation'] # , 'Partial Truncation', 'Truncation',
                #'Absorption', 'Reflection', 'Higham-Mao', 'Simple Reflection']
                # discretization schemes for SRD process

control_variate = [True]
  # use of control variate
anti_paths = [True] 
  # antithetic paths for variance reduction
moment_matching = [True] 
  # random number correction (std + mean + drift)

t_list = [1.0 / 12, 0.25, 0.5]  # maturity list
k_list = [90., 100., 110.]      # strike list

runs = 10  # number of simulation runs

np.random.seed(250000)  # set RNG seed value



#
# Function for Heston Index Process
#


def H93_index_paths(S0, r, v, row, CM):
    ''' Simulation of the Heston (1993) index process.

    Parameters
    ==========
    S0: float
        initial value
    r: NumPy array
        simulated short rate paths
    v: NumPy array
        simulated variance paths
    row: int
        row/matrix of random number array to use
    CM: NumPy array
        Cholesky matrix

    Returns
    =======
    S: NumPy array
        simulated index level paths
    '''
    sdt = math.sqrt(dt)
    S = np.zeros((M + 1, I), dtype=np.float)
    S[0] = math.log(S0)
    for t in xrange(1, M + 1, 1):
        ran = np.dot(CM, rand[:, t])
        S[t] += S[t - 1]
        S[t] += ((r[t] + r[t - 1]) / 2 - v[t] / 2) * dt
        S[t] += np.sqrt(v[t]) * ran[row] * sdt
        if momatch is True:
            S[t] -= np.mean(np.sqrt(v[t]) * ran[row] * sdt)
    return np.exp(S)


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
        rand = np.random.standard_normal((3, M + 1, I / 2))
        rand = np.concatenate((rand, -rand), 2)
    else:
        rand = np.random.standard_normal((3, M + 1, I))
    if momatch:
        rand = rand / np.std(rand)
        rand = rand - np.mean(rand)
    return rand

#
# Valuation
#
t0 = time()

results = pd.DataFrame()

tmpl_1 = '%5s | %3s | %6s | %6s | %6s | %6s | %6s | %6s | %6s | %5s | %5s'
tmpl_2 = '%4.3f | %3d ' + 7 * '| %6.3f ' + '| %5s | %5s'

for alpha in it.product(py_list, x_disc_list, m_list, paths_list,
                        control_variate, anti_paths, moment_matching):
    print '\n\n', alpha, '\n'
    (PY1, PY2), x_disc, M, I, convar, antipath, momatch = alpha
    for run in xrange(runs):  # simulation runs
        for panel in xrange(4):  # panels
            if verbose:
                print "\nResults for Panel %d\n" % (panel + 1)
                print tmpl_1 % ('T', 'K', 'V0', 'V0_LSM', 'V0_CV', 'P0',
                    'P0_MCS', 'err', 'rerr', 'acc1', 'acc2')
            # correlation matrix, cholesky decomposition
            v0, kappa_v, sigma_v, rho = para[panel]
            correlation_matrix = np.zeros((3, 3), dtype=np.float)
            correlation_matrix[0] = [1.0, rho, 0.0]
            correlation_matrix[1] = [rho, 1.0, 0.0]
            correlation_matrix[2] = [0.0, 0.0, 1.0]
            CM = np.linalg.cholesky(correlation_matrix)
            
            z = 0  # option counter
            S, r, v, h, V, matrix = 0, 0, 0, 0, 0, 0
            gc.collect()
            for T in t_list:  # times-to-maturity
                # discount factor
                B0T = B([r0, kappa_r, theta_r, sigma_r, 0.0, T])
                # average constant short rate/yield
                ra = -math.log(B0T) / T
                # time interval in years
                dt = T / M
                # pseudo-random numbers
                rand = random_number_generator(M, I)
                # short rate process paths
                r = SRD_generate_paths(x_disc, r0, kappa_r, theta_r,
                                        sigma_r, T, M, I, rand, 0, CM)
                # volatility process paths
                v = SRD_generate_paths(x_disc, v0, kappa_v, theta_v,
                                        sigma_v, T, M, I, rand, 2, CM)
                # index level process paths
                S = H93_index_paths(S0, r, v, 1, CM)
                for K in k_list:  # strikes
                    # inner value matrix
                    h = np.maximum(K - S, 0)
                    # value/cash flow matrix
                    V = np.maximum(K - S, 0)
                    for t in xrange(M - 1, 0, -1):
                        df = np.exp(-(r[t] + r[t + 1]) / 2 * dt)
                        # select only ITM paths
                        itm = np.greater(h[t], 0)
                        relevant = np.nonzero(itm)
                        rel_S = np.compress(itm, S[t])
                        no_itm = len(rel_S)
                        if no_itm == 0:
                            cv = np.zeros((I), dtype=np.float)
                        else:
                            rel_v = np.compress(itm, v[t])
                            rel_r = np.compress(itm, r[t])
                            rel_V = (np.compress(itm, V[t + 1])
                                       * np.compress(itm, df))
                            matrix = np.zeros((D + 1, no_itm), dtype=np.float)
                            matrix[10] = rel_S * rel_v * rel_r
                            matrix[9] = rel_S * rel_v
                            matrix[8] = rel_S * rel_r
                            matrix[7] = rel_v * rel_r
                            matrix[6] = rel_S ** 2
                            matrix[5] = rel_v ** 2
                            matrix[4] = rel_r ** 2
                            matrix[3] = rel_S
                            matrix[2] = rel_v
                            matrix[1] = rel_r
                            matrix[0] = 1
                            reg = np.linalg.lstsq(matrix.transpose(), rel_V)
                            cv = np.dot(reg[0], matrix)
                        erg = np.zeros((I), dtype=np.float)
                        np.put(erg, relevant, cv)
                        V[t] = np.where(h[t] > erg, h[t], V[t + 1] * df)
                    
                    # final discounting step
                    df = np.exp(-(r[0] + r[1]) / 2 * dt)
                    
                    ## European Option Values
                    C0 = H93_call_value(S0, K, T, ra, kappa_v,
                                        theta_v, sigma_v, rho, v0)

                    P0 = C0 + K * B0T - S0
                    P0_MCS = B0T * np.sum(h[-1]) / I
                    
                    x = B0T * h[-1]
                    y = V[1] * df

                    ## Control Variate Correction
                    if convar is True:
                        # statistical correlation
                        b = (np.sum((x - np.mean(x)) * (y - np.mean(y)))
                         / np.sum((x - np.mean(x)) ** 2))
                        # correction
                        y_cv = y - b * (B0T * h[-1] - P0)
                          # set b instead of 1.0
                          # to use stat. correlation
                    else:
                        y_cv = y
                    # standard error
                    SE = np.std(y_cv) / math.sqrt(I)
                    # benchmark value
                    V0 = benchmarks[panel, z]
                    # LSM control variate
                    V0_CV = max(np.sum(y_cv) / I, h[0, 0])
                    # pure LSM
                    V0_LSM = max(np.sum(y) / I, h[0, 0])
                    
                    ## Errors
                    error = V0_CV - V0
                    rel_error = error / V0
                    PY1_acc = abs(error) < PY1
                    PY2_acc = abs(rel_error) < PY2
                    res = pd.DataFrame({'timestamp': datetime.now(),
                        'runs': runs, 'PY1': PY1, 'PY2': PY2,
                        'var_disc': x_disc, 'steps': M, 'paths': I,
                        'control_variate': convar, 'anti_paths': antipath,
                        'moment_matching': momatch, 'panel': panel,
                        'maturity': T, 'strike': K, 'benchmark': V0,
                        'V0_euro': P0, 'MCS_euro': P0_MCS,
                        'LSM_pure': V0_LSM, 'LSM_convar': V0_CV,
                        'SE': SE, 'error': error, 'rel_error': rel_error,
                        'PY1_acc': PY1_acc, 'PY2_acc': PY2_acc,
                        'PY_acc': PY1_acc or PY2_acc}, 
                        index=[0,])

                    z += 1 # option counter

                    if verbose:
                        print tmpl_2 % (T, K, V0, V0_LSM, V0_CV, P0,
                            P0_MCS, error, rel_error, PY1_acc, PY2_acc)
                    
                    results = results.append(res, ignore_index=True)

if write:
    d = str(datetime.now().replace(microsecond=0))
    d = d.translate(string.maketrans("-:", "__"))
    h5 = pd.HDFStore('10_mcs/mcs_american_%s_%s_speed.h5' % (d[:10], d[11:]), 'w')
    h5['results'] = results
    h5.close()

tt = time() - t0
print "Total time in seconds     %8.2f" % tt
print "Time per option valuation %8.2f" % (tt / len(results))

