#
# Valuation of European Call Option in CRR1979 Model
# FFT Version
# A_pyt/f_CRR1979_fft.py
#
import numpy as np
from numpy.fft import fft, ifft
from c_parameters import *

# Array Generation for Index Levels
md = np.arange(M + 1)
mu = np.resize(md[-1], M + 1)
mu = u ** (mu - md)
md = d ** md
S = S0 * mu * md

# Valuation by FFT
C_T = np.maximum(S - K, 0)
Q = np.zeros(M + 1, 'd')
Q[0] = q
Q[1] = 1 - q
l = np.sqrt(M + 1)
v1 = ifft(C_T) * l
v2 = (np.sqrt(M + 1) * fft(Q) / (l * (1 + r * dt))) ** M
C_0 = fft(v1 * v2) / l

# Result Output
print("Value of European call option is %8.3f" % np.real(C_0[0]))
