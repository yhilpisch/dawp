#
# Call Option Pricing with DFT/FFT Speed Test
# 06_fou/call_fft_speed.py
#
# (c) Dr. Yves J. Hilpisch
# Derivatives Analytics with Python
#
import math
import numpy as np
from numpy.fft import fft, ifft
from convolution import revnp
from parameters import *

def call_fft_value(M):
    # Parmeter Adjustments
    dt, df, u, d, q = get_binomial_parameters(M)

    # Array Generation for Stock Prices
    mu = np.arange(M + 1)
    mu = np.resize(mu, (M + 1, M + 1))
    md = np.transpose(mu)
    mu = u ** (mu - md)
    md = d ** md
    S = S0 * mu * md

    # Valuation by FFT
    CT = np.maximum(S[:, -1] - K, 0)
    qv = np.zeros(M + 1, dtype=np.float)
    qv[0] = q
    qv[1] = 1 - q
    C0 = fft(math.exp(-r * T) * ifft(CT) * fft(qv) ** M)
