#
# Valuation of European Call Options
# in Merton's (1976) Jump Diffusion Model
# via Monte Carlo Simulation
# 08_m76/M76_valuation_MCS.py
#
# (c) Dr. Yves J. Hilpisch
# Derivatives Analytics with Python
#
import math
import numpy as np
import pandas as pd
from M76_valuation_FFT import M76_value_call_FFT
from M76_valuation_INT import M76_value_call_INT

#
# Model Parameters (from Calibration)
#
S0 = 3225.93  # EURO STOXX 50 level (30.09.2014)
T = 0.22  # shortest maturity
r = 0.005  # assumption

sigma, lamb, mu, delta = [0.113, 3.559, -0.075, 0.041]
  # from calibration


#
# Valuation by Simulation
#
seed = 100000  # seed value
M = 50  # time steps
I = 200000  # paths
disc = 2  # 1 = simple Euler; else = log Euler


def M76_generate_paths(S0, T, r, sigma, lamb, mu, delta, M, I):
    ''' Generate Monte Carlo Paths for M76 Model.
    Parameters
    ==========
    S0: float
        initial stock/index level
    K: float
        strike price
    T: float
        time-to-maturity (for t=0)
    r: float
        constant risk-free short rate
    sigma: float
        volatility factor in diffusion term
    lamb: float
        jump intensity
    mu: float
        expected jump size
    delta: float
        standard deviation of jump
    M: int
        number of time intervals
    I: int
        number of paths

    Returns
    =======
    S: array
        simulated paths
    '''
    dt = T / M
    rj = lamb * (math.exp(mu + 0.5 * delta ** 2) - 1)
    shape = (M + 1, I)
    S = np.zeros((M + 1, I), dtype=np.float)
    S[0] = S0

    np.random.seed(10000)
    rand1 = np.random.standard_normal(shape)
    rand2 = np.random.standard_normal(shape)
    rand3 = np.random.poisson(lamb * dt, shape)

    for t in xrange(1, M + 1, 1):
        if disc == 1:
            S[t] = S[t - 1] * ((1 + (r - rj) * dt) + sigma
                               * math.sqrt(dt) * rand1[t]
                               + (np.exp(mu + delta * rand2[t]) - 1)
                               * rand3[t])
        else:
            S[t] = S[t - 1] * (np.exp((r - rj - 0.5 * sigma ** 2) * dt
                               + sigma * math.sqrt(dt) * rand1[t])
                               + (np.exp(mu + delta * rand2[t]) - 1)
                               * rand3[t])
    return S

def M76_value_call_MCS(K):
    ''' Function to calculate the MCS estimator given K.

    Parameters
    ==========
    K: float
        strike price

    Returns
    =======
    call_mcs: float
        European call option Monte Carlo estimator
    '''
    return math.exp(-r * T) * np.sum(np.maximum(S[-1] - K, 0)) / I



if __name__ == '__main__':

    # Single Valuation
    S = M76_generate_paths(S0, T, r, sigma, lamb, mu, delta, M, I)
    print "Value of Call Option %8.3f" % M76_value_call_MCS(S0)

    # Value Comparisons
    strikes = np.arange(3000, 3601, 50)
    values = np.zeros((3, len(strikes)), dtype=np.float)
    z = 0
    for k in strikes:
        print "CALL STRIKE       %10.3f" % k
        print "----------------------------"
        values[0, z] = M76_value_call_INT(S0, k, T, r, sigma,
                                       lamb, mu, delta)
        print "Call Value by Int %10.3f" % values[0, z]

        values[1, z] = M76_value_call_FFT(S0, k, T, r, sigma,
                                       lamb, mu, delta)
        print "Call Value by FFT %10.3f" % values[1, z]
        print "Difference FFT/Int%10.3f" % (values[1, z] - values[0, z])
        values[2, z] = M76_value_call_MCS(k)
        print "Call Value by MCS %10.3f" % values[2, z]
        print "Difference MCS/Int%10.3f" % (values[2, z] - values[0, z])
        print "----------------------------"
        z = z + 1

    results = pd.DataFrame(values.T, index=strikes, columns=[
                                        'INT', 'FFT', 'MCS'])
    results.index.name = 'Strike'
