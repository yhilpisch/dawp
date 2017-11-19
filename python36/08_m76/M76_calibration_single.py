#
# Calibration of Merton's (1976)
# Jump Diffusion Model
# to Short Maturity Data
# 08_m76/M76_calibration_single.py
#
# (c) Dr. Yves J. Hilpisch
# Derivatives Analytics with Python
#
import math
import numpy as np
import pandas as pd
import scipy.optimize as sop
import matplotlib.pyplot as plt
import matplotlib as mpl
from M76_valuation_FFT import M76_value_call_FFT

mpl.rcParams['font.family'] = 'serif'
np.set_printoptions(suppress=True,
                    formatter={'all': lambda x: '%5.3f' % x})

#
# Market Data from www.eurexchange.com
# as of 30. September 2014
#
h5 = pd.HDFStore('08_m76/option_data.h5', 'r')
data = h5['data']  # European call & put option data (3 maturities)
h5.close()
S0 = 3225.93  # EURO STOXX 50 level
r = 0.005  # assumption

# Option Selection
tol = 0.05
options = data[(np.abs(data['Strike'] - S0) / S0) < tol]
mats = sorted(set(options['Maturity']))
options = options[options['Maturity'] == mats[0]]
options['Date'] = pd.DatetimeIndex(options['Date'])
options['Maturity'] = pd.DatetimeIndex(options['Maturity'])

#
# Error Function
#


def M76_error_function_FFT(p0):
    ''' Error function for parameter calibration in M76 Model via
    Carr-Madan (1999) FFT approach.

    Parameters
    ==========
    sigma: float
        volatility factor in diffusion term
    lamb: float
        jump intensity
    mu: float
        expected jump size
    delta: float
        standard deviation of jump

    Returns
    =======
    RMSE: float
        root mean squared error
    '''
    global i, min_RMSE
    sigma, lamb, mu, delta = p0
    if sigma < 0.0 or delta < 0.0 or lamb < 0.0:
        return 500.0
    se = []
    for row, option in options.iterrows():
        T = (option['Maturity'] - option['Date']).days / 365.
        model_value = M76_value_call_FFT(S0, option['Strike'], T,
                                         r, sigma, lamb, mu, delta)
        se.append((model_value - option['Call']) ** 2)
    RMSE = math.sqrt(sum(se) / len(se))
    min_RMSE = min(min_RMSE, RMSE)
    if i % 50 == 0:
        print('%4d |' % i, np.array(p0), '| %7.3f | %7.3f' % (RMSE, min_RMSE))
    i += 1
    return RMSE


#
# Graphical Output
#
def generate_plot(opt, options):
    #
    # Calculating Model Prices
    #
    sigma, lamb, mu, delta = opt
    options['Model'] = 0.0
    for row, option in options.iterrows():
        T = (option['Maturity'] - option['Date']).days / 365.
        options.loc[row, 'Model'] = M76_value_call_FFT(S0, option['Strike'],
                                    T, r, sigma, lamb, mu, delta)

    #
    # Plotting
    #
    options = options.set_index('Strike')
    fig, ax = plt.subplots(2, sharex=True, figsize=(8, 7))
    options[['Call', 'Model']].plot(style=['b-', 'ro'],
                    title='%s' % str(option['Maturity'])[:10], ax=ax[0])
    ax[0].set_ylabel('option values')
    ax[0].grid(True)
    xv = options.index.values
    ax[1] = plt.bar(xv - 5 / 2., options['Model'] - options['Call'],
                    width=5)
    plt.ylabel('difference')
    plt.xlim(min(xv) - 10, max(xv) + 10)
    plt.tight_layout()
    plt.grid(True)


#
# Calibration
#
if __name__ == '__main__':
    i = 0
    min_RMSE = 100.
    p0 = sop.brute(M76_error_function_FFT, ((0.10, 0.201, 0.025),
                    (0.1, 0.8, 0.1), (-0.4, 0.01, 0.1),
                    (0.00, 0.121, 0.02)), finish=None)

    opt = sop.fmin(M76_error_function_FFT, p0, xtol=0.00001,
                    ftol=0.00001, maxiter=750, maxfun=1500)
