#
# Calibration of Bakshi, Cao and Chen (1997)
# Stoch Vol Jump Model to EURO STOXX Option Quotes
# Data Source: www.eurexchange.com
# via Numerical Integration
# 11_cal/BCC97_calibration_iv.py
#
# (c) Dr. Yves J. Hilpisch
# Derivatives Analytics with Python
#
import sys
import numpy as np
import matplotlib as mpl
from scipy.optimize import fmin

sys.path.append('09_gmm')
from CIR_zcb_valuation import B
from H93_calibration import options
from BSM_imp_vol import call_option
from BCC_option_valuation import BCC_call_value
from CIR_calibration import CIR_calibration, r_list

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
S0 = 3225.93  # EURO STOXX 50 level 30.09.2014
r0 = r_list[0]  # initial short rate (Eonia 30.09.2014)

#
# Market Implied Volatilities
#
for row, option in options.iterrows():
    call = call_option(S0, option['Strike'], option['Date'],
                       option['Maturity'], option['r'], 0.15)
    options.loc[row, 'Market_IV'] = call.imp_vol(option['Call'], 0.15)

#
# Calibration Functions
#
i = 0
min_MSE = 5000.0


def BCC_iv_error_function(p0):
    ''' Error function for parameter calibration in BCC97 model via
    Lewis (2001) Fourier approach.

    Parameters
    ==========
    kappa_v: float
        mean-reversion factor
    theta_v: float
        long-run mean of variance
    sigma_v: float
        volatility of variance
    rho: float
        correlation between variance and stock/index level
    v0: float
        initial, instantaneous variance
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
    global i, min_MSE
    kappa_v, theta_v, sigma_v, rho, v0, lamb, mu, delta = p0
    if kappa_v < 0.0 or theta_v < 0.005 or sigma_v < 0.0 or \
            rho < -1.0 or rho > 1.0 or v0 < 0.0 or lamb < 0.0 or \
            mu < -.6 or mu > 0.0 or delta < 0.0:
        return 5000.0
    if 2 * kappa_v * theta_v < sigma_v ** 2:
        return 5000.0
    se = []
    for row, option in options.iterrows():
        call = call_option(S0, option['Strike'], option['Date'],
                           option['Maturity'], option['r'],
                           option['Market_IV'])
        model_value = BCC_call_value(S0, option['Strike'], option['T'],
                                     option['r'], kappa_v, theta_v, sigma_v,
                                     rho, v0, lamb, mu, delta)
        model_iv = call.imp_vol(model_value, 0.15)
        se.append(((model_iv - option['Market_IV']) * call.vega()) ** 2)
    MSE = sum(se) / len(se)
    min_MSE = min(min_MSE, MSE)
    if i % 25 == 0:
        print('%4d |' % i, np.array(p0), '| %7.3f | %7.3f' % (MSE, min_MSE))
    i += 1
    return MSE


def BCC_iv_calibration_full():
    ''' Calibrates complete BCC97 model to market implied volatilities. '''

    p0 = np.load('11_cal/opt_full.npy')

    # local, convex minimization
    opt = fmin(BCC_iv_error_function, p0,
               xtol=0.000001, ftol=0.000001,
               maxiter=450, maxfun=650)
    np.save('11_cal/opt_iv', opt)
    return opt


def BCC_calculate_model_values(p0):
    ''' Calculates all model values given parameter vector p0. '''
    kappa_v, theta_v, sigma_v, rho, v0, lamb, mu, delta = p0
    values = []
    for row, option in options.iterrows():
        model_value = BCC_call_value(S0, option['Strike'], option['T'],
                                     option['r'], kappa_v, theta_v, sigma_v,
                                     rho, v0, lamb, mu, delta)
        values.append(model_value)
    return np.array(values)
