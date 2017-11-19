#
# European Call Option Value Plot
# 02_mbv/BSM_value_plot.py
#
# (c) Dr. Yves J. Hilpisch
# Derivatives Analytics with Python
#
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['font.family'] = 'serif'

# Import Valuation Function from Chapter 5
import sys
sys.path.append('05_com')
from BSM_option_valuation import BSM_call_value

# Model and Option Parameters
K = 8000  # strike price
T = 1.0  # time-to-maturity
r = 0.025  # constant, risk-less short rate
vol = 0.2  # constant volatility

# Sample Data Generation
S = np.linspace(4000, 12000, 150)  # vector of index level values
h = np.maximum(S - K, 0)  # inner value of option
C = [BSM_call_value(S0, K, 0, T, r, vol) for S0 in S]
# calculate call option values

# Graphical Output
plt.figure()
plt.plot(S, h, 'b-.', lw=2.5, label='inner value')
# plot inner value at maturity
plt.plot(S, C, 'r', lw=2.5, label='present value')
# plot option present value
plt.grid(True)
plt.legend(loc=0)
plt.xlabel('index level $S_0$')
plt.ylabel('present value $C(t=0)$')
