#
# Delta Hedging an American Put Option in BCC97
# via Least Squares Monte Carlo (Multiple Replications)
# 13_dyh/BCC97_lsm_hedging_algorithm.py
#
# (c) Dr. Yves J. Hilpisch
# Derivatives Analytics with Python
#
import sys
sys.path.extend(['09_gmm', '11_cal', '12_val'])
import math
import numpy as np
import warnings
warnings.simplefilter('ignore')
import matplotlib as mpl
mpl.rcParams['font.family'] = 'serif'
import matplotlib.pyplot as plt
from H93_calibration import S0, kappa_r, theta_r, sigma_r, r0
from BCC97_simulation import *
from BSM_lsm_hedging_algorithm import plot_hedge_path

#
# Model Parameters
#
opt = np.load('11_cal/opt_full.npy')
kappa_v, theta_v, sigma_v, rho, v0, lamb, mu, delta = opt

#
# Simulation
#
K = S0
T = 1.0
M = 50
I = 50000
a = 1.0  # a from the interval [0.0, 2.0]
dis = 0.01  # change of S[t] in percent to estimate derivative
dt = T / M
moment_matching = True

def BCC97_lsm_put_value(S0, K, T, M, I):
    ''' Function to value American put options by LSM algorithm.

    Parameters
    ==========
    S0: float
        intial index level
    K: float
        strike of the put option
    T: float
        final date/time horizon
    M: int
        number of time steps
    I: int
        number of paths

    Returns
    =======
    V0: float
        LSM Monte Carlo estimator of American put option value
    S: NumPy array
        simulated index level paths
    r: NumPy array
        simulated short rate paths
    v: NumPy array
        simulated variance paths
    ex: NumPy array
        exercise matrix
    rg: NumPy array
        regression coefficients
    h: NumPy array
        inner value matrix
    dt: float
        length of time interval
    '''
    dt = T / M
    # Cholesky Matrix
    cho_matrix = generate_cholesky(rho)
    # Random Numbers
    rand = random_number_generator(M, I, anti_paths, moment_matching)
    # Short Rate Process Simulation
    r = SRD_generate_paths(r0, kappa_r, theta_r, sigma_r, T, M, I,
                                rand, 0, cho_matrix)
    # Variance Process Simulation
    v = SRD_generate_paths(v0, kappa_v, theta_v, sigma_v, T, M, I,
                                rand, 2, cho_matrix)
    # Index Level Process Simulation
    S = B96_generate_paths(S0, r, v, lamb, mu, delta, rand, 1, 3,
                                    cho_matrix, T, M, I, moment_matching)
    h = np.maximum(K - S, 0)  # inner value matrix
    V = np.maximum(K - S, 0)  # value/cash flow matrix
    ex = np.zeros_like(V)  # exercise matrix
    D = 10  # number of regression functions
    rg = np.zeros((M + 1, D + 1), dtype=np.float)
      # matrix for regression parameters
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
            rg[t] = np.linalg.lstsq(matrix.transpose(), rel_V)[0]
            cv = np.dot(rg[t], matrix)
        erg = np.zeros((I), dtype=np.float)
        np.put(erg, relevant, cv)
        V[t] = np.where(h[t] > erg, h[t], V[t + 1] * df)
            # value array
        ex[t] = np.where(h[t] > erg, 1, 0)
            # exercise decision
    df = np.exp(-((r[0] + r[1]) / 2) * dt)
    V0 = max(np.sum(V[1, :] * df) / I, h[0, 0])   # LSM estimator
    return V0, S, r, v, ex, rg, h, dt

def BCC97_hedge_run(p):
    ''' Implements delta hedging for a single path. '''
    #
    # Initializations
    #
    np.random.seed(50000)
    po = np.zeros(M + 1, dtype=np.float)  # vector for portfolio values
    vt = np.zeros(M + 1, dtype=np.float)  # vector for option values
    delt = np.zeros(M + 1, dtype=np.float)  # vector for deltas
    # random path selection ('real path')
    print
    print "DYNAMIC HEDGING OF AMERICAN PUT (BCC97)"
    print "---------------------------------------"
    ds = dis * S0
    V_1, S, r, v, ex, rg, h, dt = BCC97_lsm_put_value(S0 + (2 - a) * ds,
                                                   K, T, M, I)
    # 'data basis' for delta hedging
    V_2 = BCC97_lsm_put_value(S0 - a * ds, K, T, M, I)[0]
    delt[0] = (V_1 - V_2) / (2 * ds)
    V0LSM = BCC97_lsm_put_value(S0, K, T, M, I)[0]
      # initial option value for S0
    vt[0] = V0LSM  # initial option values
    po[0] = V0LSM  # initial portfolio values
    bo = V0LSM - delt[0] * S0  # initial bond position value
    print "Initial Hedge"
    print "Stocks             %8.3f" % delt[0]
    print "Bonds              %8.3f" % bo
    print "Cost               %8.3f" % (delt[0] * S0 + bo)

    print
    print "Regular Rehedges "
    print 82 * "-"
    print "step|" + 7 * " %9s|" % ('S_t', 'Port', 'Put',
                          'Diff', 'Stock', 'Bond', 'Cost')
    for t in range(1, M + 1, 1):
        if ex[t, p] == 0:
            df = math.exp((r[t, p] + r[t - 1, p]) / 2 * dt)
            if t != M:
                po[t] = delt[t - 1] * S[t, p] + bo * df
                vt[t] = BCC97_lsm_put_value(S[t, p], K, T - t * dt,
                                            M - t, I)[0]
                ds = dis * S[t, p]
                sd = S[t, p] + (2 - a) * ds  # disturbed index level
                stateV_A = [sd * v[t, p] * r[t, p],
                            sd * v[t, p],
                            sd * r[t, p],
                            v[t, p] * r[t, p],
                            sd ** 2,
                            v[t, p] ** 2,
                            r[t, p] ** 2,
                            sd,
                            v[t, p],
                            r[t, p],
                            1]
                            # state vector for S[t, p] + (2.0 - a) * dis
                stateV_A.reverse()
                V0A = max(0, np.dot(rg[t], stateV_A))
                # print V0A
                # revaluation via regression
                sd = S[t, p] - a * ds  # disturbed index level
                stateV_B = [sd * v[t, p] * r[t, p],
                            sd * v[t, p],
                            sd * r[t, p],
                            v[t, p] * r[t, p],
                            sd ** 2,
                            v[t, p] ** 2,
                            r[t, p] ** 2,
                            sd,
                            v[t, p],
                            r[t, p],
                            1]
                            # state vector for S[t, p] - a * dis
                stateV_B.reverse()
                V0B = max(0, np.dot(rg[t], stateV_B))
                # print V0B
                # revaluation via regression
                delt[t] = (V0A - V0B) / (2 * ds)
                bo = po[t] - delt[t] * S[t, p]  # bond position value
            else:
                po[t] = delt[t - 1] * S[t, p] + bo * df
                vt[t] = h[t, p]
                # inner value at final date
                delt[t] = 0.0
            print "%4d|" % t + 7 * " %9.3f|" % (S[t, p], po[t], vt[t],
                        (po[t] - vt[t]), delt[t], bo, delt[t] * S[t, p] + bo)
        else:
            po[t] = delt[t - 1] * S[t, p] + bo * df
            vt[t] = h[t, p]
            break
    errs = po - vt  # hedge errors
    print "MSE             %7.3f" % (np.sum(errs ** 2) / len(errs))
    print "Average Error   %7.3f" % (np.sum(errs) / len(errs))
    print "Total P&L       %7.3f" % np.sum(errs)
    return S[:, p], po, vt, errs, t     
