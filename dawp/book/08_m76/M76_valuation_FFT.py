#
# Valuation of European Call Options
# in Merton's (1976) Jump Diffusion Model
# via Fast Fourier Transform (FFT)
# 08_m76/M76_valuation_FFT.py
#
# (c) Dr. Yves J. Hilpisch
# Derivatives Analytics with Python
#
import math
import numpy as np
from numpy.fft import *

#
# Model Parameters
#
S0 = 100.0  # initial index level
K = 100.0  # strike level
T = 1.0  # call option maturity
r = 0.05  # constant short rate
sigma = 0.4  # constant volatility of diffusion
lamb = 1.0  # jump frequency p.a.
mu = -0.2  # expected jump size
delta = 0.1  # jump size volatility

#
# M76 Characteristic Function
#

def M76_characteristic_function(u, x0, T, r, sigma, lamb, mu, delta):
    ''' Valuation of European call option in M76 model via
    Lewis (2001) Fourier-based approach: characteristic function.

    Parameter definitions see function M76_value_call_INT. '''
    omega = x0 / T + r - 0.5 * sigma ** 2 \
                - lamb * (np.exp(mu + 0.5 * delta ** 2) - 1)
    value = np.exp((1j * u * omega - 0.5 * u ** 2 * sigma ** 2 +
            lamb * (np.exp(1j * u * mu - u ** 2 * delta ** 2 * 0.5) - 1))  * T)
    return value

#
# Valuation by FFT
#


def M76_value_call_FFT(S0, K, T, r, sigma, lamb, mu, delta):
    ''' Valuation of European call option in M76 model via
    Carr-Madan (1999) Fourier-based approach.

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
    call_value: float
        European call option present value
    '''
    k = math.log(K / S0)
    x0 = math.log(S0 / S0)
    g = 2  # factor to increase accuracy
    N = g * 4096
    eps = (g * 150.) ** -1
    eta = 2 * math.pi / (N * eps)
    b = 0.5 * N * eps - k
    u = np.arange(1, N + 1, 1)
    vo = eta * (u - 1)
    # Modificatons to Ensure Integrability
    if S0 >= 0.95 * K:  # ITM case
        alpha = 1.5
        v = vo - (alpha + 1) * 1j
        mod_char_fun = math.exp(-r * T) * M76_characteristic_function(
                                    v, x0, T, r, sigma, lamb, mu, delta) \
                / (alpha ** 2 + alpha - vo ** 2 + 1j * (2 * alpha + 1) * vo)
    else:  # OTM case
        alpha = 1.1
        v = (vo - 1j * alpha) - 1j
        mod_char_fun_1 = math.exp(-r * T) * (1 / (1 + 1j * (vo - 1j * alpha))
                                   - math.exp(r * T) / (1j * (vo - 1j * alpha))
                                   - M76_characteristic_function(
                                     v, x0, T, r, sigma, lamb, mu, delta)
                / ((vo - 1j * alpha) ** 2 - 1j * (vo - 1j * alpha)))
        v = (vo + 1j * alpha) - 1j
        mod_char_fun_2 = math.exp(-r * T) * (1 / (1 + 1j * (vo + 1j * alpha))
                                   - math.exp(r * T) / (1j * (vo + 1j * alpha))
                                   - M76_characteristic_function(
                                     v, x0, T, r, sigma, lamb, mu, delta)
                / ((vo + 1j * alpha) ** 2 - 1j * (vo + 1j * alpha)))
    
    # Numerical FFT Routine
    delt = np.zeros(N, dtype=np.float)
    delt[0] = 1
    j = np.arange(1, N + 1, 1)
    SimpsonW = (3 + (-1) ** j - delt) / 3
    if S0 >= 0.95 * K:
        fft_func = np.exp(1j * b * vo) * mod_char_fun * eta * SimpsonW
        payoff = (fft(fft_func)).real
        call_value_m = np.exp(-alpha * k) / math.pi * payoff
    else:
        fft_func = (np.exp(1j * b * vo)
                    * (mod_char_fun_1 - mod_char_fun_2)
                    * 0.5 * eta * SimpsonW)
        payoff = (fft(fft_func)).real
        call_value_m = payoff / (np.sinh(alpha * k) * math.pi)
    pos = int((k + b) / eps)
    call_value = call_value_m[pos]
    return call_value * S0

if __name__ == '__main__':
    print "Value of Call Option %8.3f" \
        % M76_value_call_FFT(S0, K, T, r, sigma, lamb, mu, delta)