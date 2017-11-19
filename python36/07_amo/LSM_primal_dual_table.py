#
# Valuation of American Options
# with Least-Squares Monte Carlo
# Primal and Dual Algorithm
# Case 1: American Put Option (APO)
# Case 2: Short Condor Spread (SCS)
# 07_amo/LSM_primal_dual_valuation.py
#
# (c) Dr. Yves J. Hilpisch
# Derivatives Analytics with Python
#
import math
import numpy as np
import pandas as pd
from time import time
from datetime import datetime
import itertools as it
import warnings
warnings.simplefilter('ignore')

t0 = time()
np.random.seed(150000)  # seed for Python RNG

# Simulation Parameters
runs = 25
write = True
otype = [1, 2]  # option type
M = [10, 25, 50]  # time steps
I1 = np.array([4]) * 4096  # replications for regression
I2 = np.array([1]) * 1024  # replications for valuation
J = [50]  # replications for nested MCS
reg = [9]  # no of basis functions
AP = [False]  # antithetic paths
MM = [True]  # moment matching of RN
ITM = [False]  # ITM paths for regression

results = pd.DataFrame()

#
# Function Definitions
#


def generate_random_numbers(I):
    ''' Function to generate I pseudo-random numbers. '''
    if AP:
        ran = np.random.standard_normal(I / 2)
        ran = np.concatenate((ran, -ran))
    else:
        ran = np.random.standard_normal(I)
    if MM:
        ran = ran - np.mean(ran)
        ran = ran / np.std(ran)
    return ran


def generate_paths(I):
    ''' Function to generate I stock price paths. '''
    S = np.zeros((M + 1, I), dtype=np.float)  # stock matrix
    S[0] = S0  # initial values
    for t in range(1, M + 1, 1):  # stock price paths
        ran = generate_random_numbers(I)
        S[t] = S[t - 1] * np.exp((r - sigma ** 2 / 2) * dt
                                 + sigma * ran * math.sqrt(dt))
    return S


def inner_values(S):
    ''' Innver value functions for American put and Short Condor Spread. '''
    if otype == 1:
        return np.maximum(40. - S, 0)
    else:
        return np.minimum(40., np.maximum(90. - S, 0)
                          + np.maximum(S - 110., 0))


def nested_monte_carlo(St, J):
    ''' Function for nested Monte Carlo simulation.

    Parameters
    ==========
    St : float
        start value for S
    J : int
        number of paths to simulate

    Returns
    =======
    paths : array
        simulated nested paths
    '''
    ran = generate_random_numbers(J)
    paths = St * np.exp((r - sigma ** 2 / 2) * dt +
                        sigma * ran * math.sqrt(dt))
    return paths


#
# Valuation
#
para = it.product(otype, M, I1, I2, J, reg, AP, MM, ITM)
count = 0
for pa in para:
    otype, M, I1, I2, J, reg, AP, MM, ITM = pa
    # General Parameters and Option Values
    if otype == 1:
        # Parameters -- American Put Option
        S0 = 36.  # initial stock level
        T = 1.0  # time-to-maturity
        r = 0.06  # short rate
        sigma = 0.2  # volatility
        V0_true = 4.48637  # American Put Option (500 steps bin. model)
    else:
        # Parameters -- Short Condor Spread
        S0 = 100.  # initial stock level
        T = 1.0  # time-to-maturity
        r = 0.05  # short rate
        sigma = 0.5  # volatility
        V0_true = 26.97705  # Short Condor Spread (500 steps bin. model)
    dt = T / M  # length of time interval
    df = math.exp(-r * dt)  # discount factor per time interval
    for j in range(runs):
        count += 1
        # regression estimation
        S = generate_paths(I1)  # generate stock price paths
        h = inner_values(S)  # inner values
        V = inner_values(S)  # value matrix
        rg = np.zeros((M + 1, reg + 1), dtype=np.float)
        # regression parameter matrix

        itm = np.greater(h, 0)  # ITM paths
        for t in range(M - 1, 0, -1):
            if ITM:
                S_itm = np.compress(itm[t] == 1, S[t])
                V_itm = np.compress(itm[t] == 1, V[t + 1])
                if len(V_itm) == 0:
                    rg[t] = 0.0
                else:
                    rg[t] = np.polyfit(S_itm, V_itm * df, reg)
            else:
                rg[t] = np.polyfit(S[t], V[t + 1] * df, reg)
                # regression at time t
            C = np.polyval(rg[t], S[t])  # continuation values
            V[t] = np.where(h[t] > C, h[t], V[t + 1] * df)
            # exercise decision

        # Simulation
        Q = np.zeros((M + 1, I2), dtype=np.float)  # martingale matrix
        U = np.zeros((M + 1, I2), dtype=np.float)  # upper bound matrix
        S = generate_paths(I2)  # generate stock price paths
        h = inner_values(S)  # inner values
        V = inner_values(S)  # value matrix

        # Primal Valuation
        for t in range(M - 1, 0, -1):
            C = np.polyval(rg[t], S[t])  # continuation values
            V[t] = np.where(h[t] > C, h[t], V[t + 1] * df)
            # exercise decision
        V0 = df * np.sum(V[1]) / I2  # LSM estimator

        # Dual Valuation
        for t in range(1, M + 1):
            for i in range(I2):
                Vt = max(h[t, i], np.polyval(rg[t], S[t, i]))
                # estimated value V(t,i)
                St = nested_monte_carlo(S[t - 1, i], J)  # nested MCS
                Ct = np.polyval(rg[t], St)  # cv from nested MCS
                ht = inner_values(St)  # iv from nested MCS
                VtJ = np.sum(np.where(ht > Ct, ht, Ct)) / len(St)
                # average of V(t,i,j)
                Q[t, i] = Q[t - 1, i] / df + (Vt - VtJ)  # "optimal" martingale
                U[t, i] = max(U[t - 1, i] / df, h[t, i] - Q[t, i])
                # high estimator values
                if t == M:
                    U[t, i] = np.maximum(U[t - 1, i] / df,
                                         np.mean(ht) - Q[t, i])
        U0 = np.sum(U[M]) / I2 * df ** M  # DUAL estimator
        AV = (V0 + U0) / 2  # average of LSM and DUAL estimator

        # output
        print("%4d | %4.1f | %48s " % (count, (time() - t0) / 60, pa),
              "| %6.3f | %6.3f | %6.3f" % (V0, U0, AV))
        # results storage
        results = results.append(pd.DataFrame({'otype': otype, 'runs': runs,
               'M': M, 'I1': I1, 'I2': I2, 'J': J, 'reg': reg, 'AP': AP,
               'MM': MM, 'ITM': ITM, 'LSM': V0, 'LSM_se': (V0 - V0_true) ** 2,
               'DUAL': U0, 'DUAL_se': (U0 - V0_true) ** 2, 'AV': AV,
               'AV_se': (AV - V0_true) ** 2}, index=[0, ]), ignore_index=True)

t1 = time()
print("Total time in min %s" % ((t1 - t0) / 60))

if write:
    h5 = pd.HDFStore('results_%s_%s.h5' % (datetime.now().date(),
                                    str(datetime.now().time())[:8]), 'w')
    h5['results'] = results
    h5.close()
