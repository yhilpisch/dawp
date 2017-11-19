#
# Calibration of Bakshi, Cao and Chen (1997)
# Stoch Vol Jump Model to EURO STOXX Option Quotes
# Data Source: www.eurexchange.com
# via Numerical Integration
# 11_cal/BCC97_calibration_short.py
#
# (c) Dr. Yves J. Hilpisch
# Derivatives Analytics with Python
#
import sys
import math
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.optimize import brute, fmin

sys.path.append('09_gmm')
from BCC_option_valuation import BCC_call_value
from CIR_calibration import CIR_calibration, r_list
from CIR_zcb_valuation import B
from H93_calibration import options

mpl.rcParams['font.family'] = 'serif'
np.set_printoptions(suppress=True,
                    formatter={'all': lambda x: '%5.3f' % x})
#
# Calibrate Short Rate Model
#
kappa_r, theta_r, sigma_r = CIR_calibration()

#
# Market Data from www.eurexchange.com
# as of 30. September 2014
#
S0 = 3225.93  # EURO STOXX 50 level
r0 = r_list[0]  # initial short rate (Eonia 30.09.2014)
#
# Option Selection
#
mats = sorted(set(options['Maturity']))
# only shortest maturity
options = options[options['Maturity'] == mats[0]]

#
# Initial Parameter Guesses
#
# from H93 model calibration
kappa_v, theta_v, sigma_v, rho, v0 = np.load('11_cal/opt_sv.npy')


#
# Calibration Functions
#
i = 0
min_MSE = 5000.0
local_opt = False


def BCC_error_function(p0):
    ''' Error function for parameter calibration in M76 Model via
    Carr-Madan (1999) FFT approach.

    Parameters
    ==========
    lamb: float
        jump intensity
    mu: float
        expected jump size
    delta: float
        standard deviation of jump

    Returns
    =======
    MSE: float
        mean squared error
    '''
    global i, min_MSE, local_opt, opt1
    lamb, mu, delta = p0
    if lamb < 0.0 or mu < -0.6 or mu > 0.0 or delta < 0.0:
        return 5000.0
    se = []
    for row, option in options.iterrows():
        model_value = BCC_call_value(S0, option['Strike'], option['T'],
                                     option['r'], kappa_v, theta_v, sigma_v,
                                     rho, v0, lamb, mu, delta)
        se.append((model_value - option['Call']) ** 2)
    MSE = sum(se) / len(se)
    min_MSE = min(min_MSE, MSE)
    if i % 25 == 0:
        print('%4d |' % i, np.array(p0), '| %7.3f | %7.3f' % (MSE, min_MSE))
    i += 1
    if local_opt:
        penalty = np.sqrt(np.sum((p0 - opt1) ** 2)) * 1
        return MSE + penalty
    return MSE

#
# Calibration
#


def BCC_calibration_short():
    ''' Calibrates jump component of BCC97 model to market quotes. '''
    # first run with brute force
    # (scan sensible regions)
    opt1 = 0.0
    opt1 = brute(BCC_error_function,
                 ((0.0, 0.51, 0.1),  # lambda
                  (-0.5, -0.11, 0.1),  # mu
                  (0.0, 0.51, 0.25)),  # delta
                 finish=None)

    # second run with local, convex minimization
    # (dig deeper where promising)
    opt2 = fmin(BCC_error_function, opt1,
                xtol=0.0000001, ftol=0.0000001,
                maxiter=550, maxfun=750)
    np.save('11_cal/opt_jump', np.array(opt2))
    return opt2


def BCC_jump_calculate_model_values(p0):
    ''' Calculates all model values given parameter vector p0. '''
    lamb, mu, delta = p0
    values = []
    for row, option in options.iterrows():
        T = (option['Maturity'] - option['Date']).days / 365.
        B0T = B([kappa_r, theta_r, sigma_r, r0, T])
        r = -math.log(B0T) / T
        model_value = BCC_call_value(S0, option['Strike'], T, r,
                                     kappa_v, theta_v, sigma_v, rho, v0,
                                     lamb, mu, delta)
        values.append(model_value)
    return np.array(values)

#
# Graphical Results Output
#


def plot_calibration_results(p0):
    options['Model'] = BCC_jump_calculate_model_values(p0)
    plt.figure(figsize=(8, 6))
    plt.subplot(211)
    plt.title('Maturity %s' % str(options['Maturity'].iloc[0])[:10])
    plt.ylabel('option values')
    plt.plot(options.Strike, options.Call, 'b', label='market')
    plt.plot(options.Strike, options.Model, 'ro', label='model')
    plt.legend(loc=0)
    plt.axis([min(options.Strike) - 10, max(options.Strike) + 10,
              min(options.Call) - 10, max(options.Call) + 10])
    plt.subplot(212)
    wi = 5.0
    diffs = options.Model.values - options.Call.values
    plt.bar(options.Strike.values - wi / 2, diffs, width=wi)
    plt.ylabel('difference')
    plt.axis([min(options.Strike) - 10, max(options.Strike) + 10,
              min(diffs) * 1.1, max(diffs) * 1.1])
    plt.tight_layout()
