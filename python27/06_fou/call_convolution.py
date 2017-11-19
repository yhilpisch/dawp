#
# Call Option Pricing with Circular Convolution (Simple)
# 06_fou/call_convolution.py
#
# (c) Dr. Yves J. Hilpisch
# Derivatives Analytics with Python
#
import math
import numpy as np
from convolution import revnp, convolution

# Parameter Definitions
M = 4  # number of time steps
dt = 1.0 / M  # length of time interval
r = 0.05  # constant short rate
C = [49.18246976, 22.14027582, 0, 0, 0]  # call payoff at maturity
q = 0.537808372  # martingale probability
qv = np.array([q, 1 - q, 0, 0, 0])  # probabilitiy vector filled with zeros

# Calculation
V = np.zeros((M + 1, M + 1), dtype=np.float)
V[M] = C

for t in range(M - 1, -1, -1):
    V[t] = convolution(V[t + 1], revnp(qv)) * math.exp(-r * dt)

print "Value of the Call Option %8.3f" % V[0, 0]

