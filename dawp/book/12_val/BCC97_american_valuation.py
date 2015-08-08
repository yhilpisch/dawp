#
# Valuation of American Options in BCC97 Model
# by Least-Squares Monte Carlo Algorithm
# 12_val/BCC97_american_valuation.py
#
# (c) Dr. Yves J. Hilpisch
# Derivatives Analytics with Python
#
import sys
sys.path.extend(['09_gmm', '10_mcs'])
import math
from BCC_option_valuation import *
from CIR_zcb_valuation_gen import B
from BCC97_simulation import *

#
# Additional Parameters
#
D = 10  # number of basis functions
t_list = [1 / 12., 0.5, 1.0, 1.5, 2.0, 3.0]
k_list = [3050, 3225, 3400]

#
# LSM Valuation Function
#
def BCC97_lsm_valuation(S, r, v, K, T, M, I):
    ''' Function to value American put options by LSM algorithm.

    Parameters
    ==========
    S: NumPy array
        simulated index level paths
    r: NumPy array
        simulated short rate paths
    v: NumPy array
        simulated variance paths
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
    LSM_value: float
        LSM Monte Carlo estimator of American put option value
    '''
    dt = T / M
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
            # exercise decision
    df = np.exp(-((r[0] + r[1]) / 2) * dt)
    LSM_value = max(np.sum(V[1, :] * df) / I, h[0, 0])   # LSM estimator
    return LSM_value

#
# Valuation for Different Strikes & Maturities
#
def lsm_compare_values(M0=50, I=50000):
    results = []
    for T in t_list:
        #
        # Simulation
        #
        M = int(M0 * T)
        cho_matrix = generate_cholesky(rho)
        rand = random_number_generator(M, I, anti_paths, moment_matching)
        r = SRD_generate_paths(r0, kappa_r, theta_r, sigma_r, T, M, I,
                                rand, 0, cho_matrix)
        v = SRD_generate_paths(v0, kappa_v, theta_v, sigma_v, T, M, I,
                                rand, 2, cho_matrix)
        S = B96_generate_paths(S0, r, v, lamb, mu, delta, rand, 1, 3,
                                    cho_matrix, T, M, I, moment_matching)
        for K in k_list:
            #
            # Valuation
            #
            h = np.maximum(S[-1] - K, 0)
            B0T = B([r0, kappa_r, theta_r, sigma_r, 0.0, T])
            V0_lsm = BCC97_lsm_valuation(S, r, v, K, T, M, I)
                # LSM estimator
            #
            # European Call Option via Fourier
            #
            ra = -math.log(B0T) / T  # average short rate/yield
            C0 = BCC_call_value(S0, K, T, ra, kappa_v, theta_v, sigma_v,
                                rho, v0, lamb, mu, delta)
            P0 = C0 + K * B0T - S0

            results.append((T, K, P0, V0_lsm, V0_lsm - P0))

    print " %6s | %6s | %7s | %7s | %7s" % ('T', 'K', 'P0', 'LSM', 'DIFF')
    for res in results:
        print " %6.3f | %6d | %7.3f | %7.3f | %7.3f" % res
