#
# Valuation of European Call and Put Options
# Under Stochastic Volatility and Jumps
# 09_gmm/BCC_option_valuation.py
#
# (c) Dr. Yves J. Hilpisch
# Derivatives Analytics with Python
#
import numpy as np
from scipy.integrate import quad
from CIR_zcb_valuation import B
import warnings
warnings.simplefilter('ignore')

#
# Example Parameters B96 Model
#
## H93 Parameters
kappa_v = 1.5
theta_v = 0.02
sigma_v = 0.15
rho = 0.1
v0 = 0.01

## M76 Parameters
lamb = 0.25
mu = -0.2
delta = 0.1
sigma = np.sqrt(v0)

## General Parameters
S0 = 100.0
K = 100.0
T = 1.0
r = 0.05

#
# Valuation by Integration
#


def BCC_call_value(S0, K, T, r, kappa_v, theta_v, sigma_v, rho, v0,
                         lamb, mu, delta):
    ''' Valuation of European call option in B96 Model via Lewis (2001)
    Fourier-based approach.

    Parameters
    ==========
    S0: float
        initial stock/index level
    K: float
        strike price
    T: float
        time-to-maturity (for t=0)
    r: float
        constant risk-free short rate
    kappa_v: float
        mean-reversion factor
    theta_v: float
        long-run mean of variance
    sigma_v: float
        volatility of variance
    rho: float
        correlation between variance and stock/index level
    v0: float
        initial level of variance
    lamb: float
        jump intensity
    mu: float
        expected jump size
    delta: float
        standard deviation of jump

    Returns
    =======
    call_value: float
        present value of European call option

    '''
    int_value = quad(lambda u: BCC_int_func(u, S0, K, T, r, kappa_v, theta_v, 
                sigma_v, rho, v0, lamb, mu, delta), 0, np.inf, limit=250)[0]
    call_value = max(0, S0 - np.exp(-r * T) * np.sqrt(S0 * K)
                            / np.pi * int_value)
    return call_value


def H93_call_value(S0, K, T, r, kappa_v, theta_v, sigma_v, rho, v0):
    ''' Valuation of European call option in H93 model via Lewis (2001)
    Fourier-based approach.

    Parameters
    ==========
    S0: float
        initial stock/index level
    K: float
        strike price
    T: float
        time-to-maturity (for t=0)
    r: float
        constant risk-free short rate
    kappa_v: float
        mean-reversion factor
    theta_v: float
        long-run mean of variance
    sigma_v: float
        volatility of variance
    rho: float
        correlation between variance and stock/index level
    v0: float
        initial level of variance

    Returns
    =======
    call_value: float
        present value of European call option

    '''
    int_value = quad(lambda u: H93_int_func(u, S0, K, T, r, kappa_v,
                        theta_v, sigma_v, rho, v0), 0, np.inf, limit=250)[0]
    call_value = max(0, S0 - np.exp(-r * T) * np.sqrt(S0 * K)
                            / np.pi * int_value)
    return call_value


def M76_call_value(S0, K, T, r, v0, lamb, mu, delta):
    ''' Valuation of European call option in M76 model via Lewis (2001)
    Fourier-based approach.

    Parameters
    ==========
    S0: float
        initial stock/index level
    K: float
        strike price
    T: float
        time-to-maturity (for t=0)
    r: float
        constant risk-free short rate
    lamb: float
        jump intensity
    mu: float
        expected jump size
    delta: float
        standard deviation of jump

    Returns
    =======
    call_value: float
        present value of European call option
    '''
    sigma = np.sqrt(v0)
    int_value = quad(lambda u: M76_int_func_sa(u, S0, K, T, r,
                        sigma, lamb, mu, delta), 0, np.inf, limit=250)[0]
    call_value = max(0, S0 - np.exp(-r * T) * np.sqrt(S0 * K)
                            / np.pi * int_value)
    return call_value


#
# Integration Functions
#


def BCC_int_func(u, S0, K, T, r, kappa_v, theta_v, sigma_v, rho, v0,
                            lamb, mu, delta):
    ''' Valuation of European call option in BCC97 model via Lewis (2001)
    Fourier-based approach: integration function.

    Parameter definitions see function BCC_call_value.'''
    char_func_value = BCC_char_func(u - 1j * 0.5, T, r, kappa_v, theta_v, 
                        sigma_v, rho, v0, lamb, mu, delta)
    int_func_value = 1 / (u ** 2 + 0.25) \
            * (np.exp(1j * u * np.log(S0 / K)) * char_func_value).real
    return int_func_value


def H93_int_func(u, S0, K, T, r, kappa_v, theta_v, sigma_v, rho, v0):
    ''' Valuation of European call option in H93 model via Lewis (2001)
    Fourier-based approach: integration function.

    Parameter definitions see function H93_call_value.'''
    char_func_value = H93_char_func(u - 1j * 0.5, T, r, kappa_v,
                                    theta_v, sigma_v, rho, v0)
    int_func_value = 1 / (u ** 2 + 0.25) \
            * (np.exp(1j * u * np.log(S0 / K)) * char_func_value).real
    return int_func_value


def M76_int_func_sa(u, S0, K, T, r, sigma, lamb, mu, delta):
    ''' Valuation of European call option in M76 model via Lewis (2001)
    Fourier-based approach: integration function.

    Parameter definitions see function M76_call_value.'''
    char_func_value = M76_char_func_sa(u - 0.5 * 1j, T, r, sigma,
                                        lamb, mu, delta)
    int_func_value = 1 / (u ** 2 + 0.25) \
            * (np.exp(1j * u * np.log(S0 / K)) * char_func_value).real
    return int_func_value

#
# Characteristic Functions
#


def BCC_char_func(u, T, r, kappa_v, theta_v, sigma_v, rho, v0,
                    lamb, mu, delta):
    ''' Valuation of European call option in BCC97 model via Lewis (2001)
    Fourier-based approach: characteristic function.

    Parameter definitions see function BCC_call_value.'''
    BCC1 = H93_char_func(u, T, r, kappa_v, theta_v, sigma_v, rho, v0)
    BCC2 = M76_char_func(u, T, lamb, mu, delta)
    return BCC1 * BCC2


def H93_char_func(u, T, r, kappa_v, theta_v, sigma_v, rho, v0):
    ''' Valuation of European call option in H93 model via Lewis (2001)
    Fourier-based approach: characteristic function.

    Parameter definitions see function BCC_call_value.'''
    c1 = kappa_v * theta_v
    c2 = -np.sqrt((rho * sigma_v * u * 1j - kappa_v)
            ** 2 - sigma_v ** 2 * (-u * 1j - u ** 2))
    c3 = (kappa_v - rho * sigma_v * u * 1j + c2) \
          / (kappa_v - rho * sigma_v * u * 1j - c2)
    H1 = (r * u * 1j * T + (c1 / sigma_v ** 2)
          * ((kappa_v - rho * sigma_v * u * 1j + c2) * T
                - 2 * np.log((1 - c3 * np.exp(c2 * T)) / (1 - c3))))
    H2 = ((kappa_v - rho * sigma_v * u * 1j + c2) / sigma_v ** 2
          * ((1 - np.exp(c2 * T)) / (1 - c3 * np.exp(c2 * T))))
    char_func_value = np.exp(H1 + H2 * v0)
    return char_func_value


def M76_char_func(u, T, lamb, mu, delta):
    ''' Valuation of European call option in M76 model via Lewis (2001)
    Fourier-based approach: characteristic function.

    Parameter definitions see function M76_call_value.'''
    omega = -lamb * (np.exp(mu + 0.5 * delta ** 2) - 1)
    char_func_value = np.exp((1j * u * omega + lamb
            * (np.exp(1j * u * mu - u ** 2 * delta ** 2 * 0.5) - 1)) * T)
    return char_func_value


def M76_char_func_sa(u, T, r, sigma, lamb, mu, delta):
    ''' Valuation of European call option in M76 model via Lewis (2001)
    Fourier-based approach: characteristic function "jump component".

    Parameter definitions see function M76_call_value.'''
    omega = r - 0.5 * sigma ** 2 - lamb * (np.exp(mu + 0.5 * delta ** 2) - 1)
    char_func_value = np.exp((1j * u * omega - 0.5 * u ** 2 * sigma ** 2
                + lamb * (np.exp(1j * u * mu - u ** 2 * delta ** 2 * 0.5)
                    - 1)) * T)
    return char_func_value

if __name__ == '__main__':
    #
    # Example Parameters CIR85 Model
    #
    kappa_r, theta_r, sigma_r, r0, T = 0.3, 0.04, 0.1, 0.04, T
    B0T = B([kappa_r, theta_r, sigma_r, r0, T])  # discount factor
    r = -np.log(B0T) / T

    #
    # Example Values
    #
    print "M76 Value   %10.4f" \
        % M76_call_value(S0, K, T, r, v0, lamb, mu, delta)
    print "H93 Value   %10.4f" \
        % H93_call_value(S0, K, T, r, kappa_v, theta_v, sigma_v, rho, v0)
    print "BCC97 Value %10.4f" \
        % BCC_call_value(S0, K, T, r, kappa_v, theta_v,
                           sigma_v, rho, v0, lamb, mu, delta)
