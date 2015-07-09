#
# Retrieving Financial Data from the Web and
# Doing Data Analytics with pandas
# A_pyt/n_pandas.py
#
import math
import numpy as np
import pandas as pd
import pandas.io.data as web
import matplotlib.pyplot as plt

#
# 1. Data Gathering
#
DAX = web.DataReader('^GDAXI', data_source='yahoo',
                     start='1/1/2005', end='28/11/2014')
    # reads DAX data from Yahoo Finance

#
# 2. Data Analysis
#
DAX['Returns'] = np.log(DAX['Close'] / DAX['Close'].shift(1))
    # daily log returns

#
# 3. Generating Plots
#
plt.figure(figsize=(7, 5))
plt.subplot(211)
DAX['Adj Close'].plot(grid=True)
plt.title('DAX Index')
plt.subplot(212)
DAX['Returns'].plot(grid=True)
plt.title('log returns')
plt.tight_layout()

#
# 4. Numerical Methods
#
# Market Parameters
S0 = DAX['Close'][-1]  # start value of DAX for simulation
vol = np.std(DAX['Returns']) * math.sqrt(252)
  # historical, annualized volatility of DAX
r = 0.01  # constant risk-free short rate

# Option Parameters
K = 10000.  # strike price of the option to value
T = 1.0  # time-to-maturity of the option

# Simulation Parameters
M = 50  # number of time steps
dt = T / M  # length of time interval
I = 10000  # number of paths to simulate
np.random.seed(5000)  # fixed seed value

# Simulation
S = np.zeros((M + 1, I), dtype=np.float)  # array for simulated DAX levels
S[0] = S0  # initial values
for t in xrange(1, M + 1):
    ran = np.random.standard_normal(I)  # pseudo-random numbers
    S[t] = S[t - 1] * np.exp((r - vol ** 2 / 2) * dt
                                + vol * math.sqrt(dt) * ran)
        # difference equation to simulate DAX levels step-by-step
        # NumPy vectorization over all simulated paths

# Valuation
V0 = math.exp(-r * T) * np.sum(np.maximum(S[-1] - K, 0)) / I  # MCS estimator
print "MCS call value estimate is %8.3f" % V0

#
# 5. Data Storage
#
h5file = pd.HDFStore('DAX_data.h5')  # open HDF5 file as database
h5file['DAX'] = DAX  # write pandas.DataFrame DAX into HDFStore
h5file.close()  # close file
