#
# Valuation of American Options
# with the Cox-Ross-Rubinstein Model
# Primal Algorithm
# Case 1: American Put Option (APO)
# Case 2: Short Condor Spread (SCS)
# 07_amo/CRR_american_options.py
#
# (c) Dr. Yves J. Hilpisch
# Derivatives Analytics with Python
#
import math
import numpy as np

# General Parameters and Option Values


def set_parameters(otype, M):
    ''' Sets parameters depending on valuation case.

    Parameters
    ==========
    otype : int
        option type
        1 = American put option
        2 = Short Condor Spread
    '''
    if otype == 1:
        # Parameters -- American Put Option
        S0 = 36.  # initial stock level
        T = 1.0  # time-to-maturity
        r = 0.06  # short rate
        sigma = 0.2  # volatility

    elif otype == 2:
        # Parameters -- Short Condor Spread
        S0 = 100.  # initial stock level
        T = 1.0  # time-to-maturity
        r = 0.05  # short rate
        sigma = 0.5  # volatility

    else:
        raise ValueError('Option type not known.')

    # Numerical Parameters
    dt = T / M  # time interval
    df = math.exp(-r * dt)  # discount factor
    u = math.exp(sigma * math.sqrt(dt))  # up-movement
    d = 1 / u  # down-movement
    q = (math.exp(r * dt) - d) / (u - d)  # martingale probability

    return S0, T, r, sigma, M, dt, df, u, d, q


def inner_value(S, otype):
    ''' Inner value functions for American put option and short condor spread
    option with American exercise.

    Parameters
    ==========
    otype : int
        option type
        1 = American put option
        2 = Short Condor Spread
    '''
    if otype == 1:
        return np.maximum(40. - S, 0)
    elif otype == 2:
        return np.minimum(40., np.maximum(90. - S, 0) + 
                          np.maximum(S - 110., 0))
    else:
        raise ValueError('Option type not known.')


def CRR_option_valuation(otype, M=500):
    S0, T, r, sigma, M, dt, df, u, d, q = set_parameters(otype, M)
    # Array Generation for Stock Prices
    mu = np.arange(M + 1)
    mu = np.resize(mu, (M + 1, M + 1))
    md = np.transpose(mu)
    mu = u ** (mu - md)
    md = d ** md
    S = S0 * mu * md

    # Valuation by Backwards Induction
    h = inner_value(S, otype)  # innver value matrix
    V = inner_value(S, otype)  # value matrix
    C = np.zeros((M + 1, M + 1), dtype=np.float)  # continuation values
    ex = np.zeros((M + 1, M + 1), dtype=np.float)  # exercise matrix

    z = 0
    for i in range(M - 1, -1, -1):
        C[0:M - z, i] = (q * V[0:M - z, i + 1] +
                         (1 - q) * V[1:M - z + 1, i + 1]) * df
        V[0:M - z, i] = np.where(h[0:M - z, i] > C[0:M - z, i],
                                 h[0:M - z, i], C[0:M - z, i])
        ex[0:M - z, i] = np.where(h[0:M - z, i] > C[0:M - z, i], 1, 0)
        z += 1
    return V[0, 0]
