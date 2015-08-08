#
# Valuation of European Options in BCC97 Model
# by Monte Carlo Simulation
# 12_val/BCC97_valuation_comparison.py
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
# Parameters
#
t_list = [1 / 12., 0.5, 1.0, 1.5, 2.0, 3.0]
k_list = [3050, 3225, 3400]

#
# Valuation for Different Strikes & Maturities
#
def compare_values(M0=50, I=50000):
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
            V0_mcs = B0T * np.sum(h) / I # MCS estimator
            #
            # European Call Option via Fourier
            #
            ra = -math.log(B0T) / T  # average short rate/yield
            C0 = BCC_call_value(S0, K, T, ra, kappa_v, theta_v, sigma_v,
                                rho, v0, lamb, mu, delta)
            
            results.append((T, K, C0, V0_mcs, V0_mcs - C0))

    print " %6s | %6s | %7s | %7s | %7s" % ('T', 'K', 'C0', 'MCS', 'DIFF')
    for res in results:
        print " %6.3f | %6d | %7.3f | %7.3f | %7.3f" % res