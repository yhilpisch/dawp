#
# Valuation of Zero-Coupon Bonds
# in Cox-Ingersoll-Ross (1985) Model
# 09_gmm/CIR_zcb_valuation.py
#
# (c) Dr. Yves J. Hilpisch
# Derivatives Analytics with Python
#
import math
import numpy as np

#
# Example Parameters CIR85 Model
#
kappa_r, theta_r, sigma_r, r0, T = 0.3, 0.04, 0.1, 0.04, 1.0

#
# Zero-Coupon Bond Valuation Formula
#


def gamma(kappa_r, sigma_r):
    ''' Help Function. '''
    return math.sqrt(kappa_r ** 2 + 2 * sigma_r ** 2)


def b1(alpha):
    ''' Help Function. '''
    kappa_r, theta_r, sigma_r, r0, T = alpha
    g = gamma(kappa_r, sigma_r)
    return (((2 * g * math.exp((kappa_r + g) * T / 2)) /
             (2 * g + (kappa_r + g) * (math.exp(g * T) - 1))) **
            (2 * kappa_r * theta_r / sigma_r ** 2))


def b2(alpha):
    ''' Help Function. '''
    kappa_r, theta_r, sigma_r, r0, T = alpha
    g = gamma(kappa_r, sigma_r)
    return ((2 * (math.exp(g * T) - 1)) /
            (2 * g + (kappa_r + g) * (math.exp(g * T) - 1)))


def B(alpha):
    ''' Function to value unit zero-coupon bonds in Cox-Ingersoll-Ross (1985)
    model.

    Parameters
    ==========
    r0: float
        initial short rate
    kappa_r: float
        mean-reversion factor
    theta_r: float
        long-run mean of short rate
    sigma_r: float
        volatility of short rate
    T: float
        time horizon/interval

    Returns
    =======
    zcb_value: float
        zero-coupon bond present value
    '''
    b_1 = b1(alpha)
    b_2 = b2(alpha)
    kappa_r, theta_r, sigma_r, r0, T = alpha
    return b_1 * math.exp(-b_2 * r0)


if __name__ == '__main__':
    #
    # Example Valuation
    #
    B0T = B([kappa_r, theta_r, sigma_r, r0, T])
    # discount factor, ZCB value
    print("ZCB Value   %10.4f" % B0T)
