#
# Valuation of European Call Option in CRR1979 Model
# Loop Version (= C-like Iterations)
# A_pyt/d_CRR1979_loop.py
#
import numpy as np
from c_parameters import *

# Array Initialization for Index Levels
S = np.zeros((M + 1, M + 1), dtype=np.float)  # index level array
S[0, 0] = S0
z = 0
for j in range(1, M + 1, 1):
    z += 1
    for i in range(z + 1):
        S[i, j] = S[0, 0] * (u ** j) * (d ** (i * 2))

# Array Initialization for Inner Values
iv = np.zeros((M + 1, M + 1), dtype=np.float)  # inner value array
z = 0
for j in range(0, M + 1, 1):
    for i in range(z + 1):
        iv[i, j] = round(max(S[i, j] - K, 0), 8)
    z += 1

# Valuation by Risk-Neutral Discounting
pv = np.zeros((M + 1, M + 1), dtype=np.float)  # present value array
pv[:, M] = iv[:, M]  # initialize last time step
z = M + 1
for j in range(M - 1, -1, -1):
    z -= 1
    for i in range(z):
        pv[i, j] = (q * pv[i, j + 1] + (1 - q) * pv[i + 1, j + 1]) * df

# Result Output
print("Value of European call option is %8.3f" % pv[0, 0])
