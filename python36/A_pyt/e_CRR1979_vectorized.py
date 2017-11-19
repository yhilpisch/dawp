#
# Valuation of European Call Option in CRR1979 Model
# Vectorized Version (= NumPy-level Iterations)
# A_pyt/e_CRR1979_vectorized.py
#
import numpy as np
from c_parameters import *

# Array Initialization for Index Levels
mu = np.arange(M + 1)
mu = np.resize(mu, (M + 1, M + 1))
md = np.transpose(mu)
mu = u ** (mu - md)
md = d ** md
S = S0 * mu * md

# Valuation by Risk-Neutral Discounting
pv = np.maximum(S - K, 0)  # present value array initialized with inner values
z = 0
for i in range(M - 1, -1, -1):  # backwards induction
    pv[0:M - z, i] = (q * pv[0:M - z, i + 1] +
                      (1 - q) * pv[1:M - z + 1, i + 1]) * df
    z += 1

# Result Output
print("Value of European call option is %8.3f" % pv[0, 0])
