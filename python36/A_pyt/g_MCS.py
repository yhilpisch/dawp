#
# Valuation of European Call Option
# via Monte Carlo Simulation
# A_pyt/g_MCS.py
#
import numpy as np
import matplotlib.pyplot as plt
from c_parameters import *

# Valuation via MCS
I = 100000  # number of simulated values for S_T
rand = np.random.standard_normal(I)  # generate pseude-random numbers
# simulate I values for S_T
ST = S0 * np.exp((r - 0.5 * vola ** 2) * T + np.sqrt(T) * vola * rand)
pv = np.sum(np.maximum(ST - K, 0) * np.exp(-r * T)) / I  # MCS estimator

# Result Output
print("Value of European call option is %8.3f" % pv)

# Graphical Output
plt.figure()
plt.hist(ST, 100)
plt.xlabel('index level at T')
plt.ylabel('frequency')
