#
# Call Option Pricing with Discrete Fourier Transforms (DFT/FFT)
# 06_fou/call_fft_pricing.py
#
# (c) Dr. Yves J. Hilpisch
# Derivatives Analytics with Python
#
import math
import numpy as np
from numpy.fft import fft, ifft
from convolution import revnp
from parameters import *

# Parmeter Adjustments
M = 3  # number of time steps
dt, df, u, d, q = get_binomial_parameters(M)

# Array Generation for Stock Prices
mu = np.arange(M + 1)
mu = np.resize(mu, (M + 1, M + 1))
md = np.transpose(mu)
mu = u ** (mu - md)
md = d ** md
S = S0 * mu * md

# Valuation by fft
CT = np.maximum(S[:, -1] - K, 0)
qv = np.zeros(M + 1, dtype=np.float)
qv[0] = q
qv[1] = 1 - q
C0_a = fft(math.exp(-r * T) * ifft(CT) * ((M + 1) * ifft(revnp(qv))) ** M)
C0_b = fft(math.exp(-r * T) * ifft(CT) * fft(qv) ** M)
C0_c = ifft(math.exp(-r * T) * fft(CT) * fft(revnp(qv)) ** M)

# Results Outpu
print "Value of European option is %8.3f" % np.real(C0_a[0])
print "Value of European option is %8.3f" % np.real(C0_b[0])
print "Value of European option is %8.3f" % np.real(C0_c[0])
