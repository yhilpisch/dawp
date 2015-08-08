#
# Calibration of Merton's (1976)
# Jump Diffusion Model
# via Fast Fourier Transform
# 08_m76/M76_calibration_FFT.py
#
# (c) Dr. Yves J. Hilpisch
# Derivatives Analytics with Python
#
import math
import numpy as np
np.set_printoptions(suppress=True,
                    formatter={'all': lambda x: '%5.3f' % x})
import pandas as pd
import scipy.optimize as sop
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['font.family'] = 'serif'
from M76_valuation_FFT import M76_value_call_FFT

#
# Market Data from www.eurexchange.com
# as of 30. September 2014
#
h5 = pd.HDFStore('08_m76/option_data.h5', 'r')
data = h5['data']  # European call & put option data (3 maturities)
h5.close()
S0 = 3225.93  # EURO STOXX 50 level
r = 0.0005  # ECB base rate

# Option Selection
tol = 0.02
options = data[(np.abs(data['Strike'] - S0) / S0) < tol]
     
#
# Error Function
#

def M76_error_function_FFT(p0):
    ''' Error Function for parameter calibration in M76 Model via
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
        print '%4d |' % i, np.array(p0), '| %7.3f | %7.3f' % (RMSE, min_RMSE)
    i += 1
    return RMSE

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
    mats = sorted(set(options['Maturity']))
    options = options.set_index('Strike')
    for i, mat in enumerate(mats):
        options[options['Maturity'] == mat][['Call', 'Model']].\
                plot(style=['b-', 'ro'], title='%s' % str(mat)[:10],
                grid=True)
        plt.ylabel('option value')
        plt.savefig('../images/08_m76/M76_calibration_3_%s.pdf' % i)

if __name__ == '__main__':
    #
    # Calibration
    #
    i = 0  # counter initialization
    min_RMSE = 100  # minimal RMSE initialization
    p0 = sop.brute(M76_error_function_FFT, ((0.075, 0.201, 0.025),
                    (0.10, 0.401, 0.1), (-0.5, 0.01, 0.1),
                    (0.10, 0.301, 0.1)), finish=None)

    # p0 = [0.15, 0.2, -0.3, 0.2]
    opt = sop.fmin(M76_error_function_FFT, p0,
                    maxiter=500, maxfun=750,
                    xtol=0.000001, ftol=0.000001)
