#
# Call Option Pricing with Circular Convolution (General)
# 06_fou/call_convolution_general.py
#
# (c) Dr. Yves J. Hilpisch
# Derivatives Analytics with Python
#
import numpy as np
from convolution import revnp, convolution
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

# Valuation
V = np.maximum(S - K, 0)
qv = np.zeros((M + 1), dtype=np.float)
qv[0] = q
qv[1] = 1 - q
for t in range(M - 1, -1, -1):
    V[:, t] = convolution(V[:, t + 1], revnp(qv)) * df

print "Value of the Call Option %8.3f" % V[0, 0]
