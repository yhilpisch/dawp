#
# Two Financial Option Classes
# A_pyt/l_CLASS.py
#
#
import math
import scipy.stats as scs

# Class Definitions


class Option(object):
    ''' Black-Scholes-Merton European call option class.

    Attributes
    ==========
    S0: float
        initial index level
    K: float
        strike price
    T: float
        time-to-maturity
    r: float
        constant short rate
    vola: float
        constant volatility factor
    '''

    def __init__(self, S0, K, T, r, vola):
        ''' Initialization of Object. '''
        self.S0 = float(S0)
        self.K = K
        self.T = T
        self.r = r
        self.vola = vola

    def d1(self):
        ''' Helper function. '''
        d1 = ((math.log(self.S0 / self.K) +
               (self.r + 0.5 * self.vola ** 2) * self.T) /
              (self.vola * math.sqrt(self.T)))
        return d1

    def value(self):
        ''' Method to value option. '''
        d1 = self.d1()
        d2 = d1 - self.vola * math.sqrt(self.T)
        call_value = (self.S0 * scs.norm.cdf(d1, 0.0, 1.0) -
                      self.K * math.exp(-self.r * self.T) *
                      scs.norm.cdf(d2, 0.0, 1.0))
        return call_value


class OptionVega(Option):
    ''' Black-Scholes-Merton class for Vega of European call option. '''

    def vega(self):
        ''' Method to calculate the Vega of the European call option. '''
        d1 = self.d1()
        vega = self.S0 * scs.norm.cdf(d1, 0.0, 1.0) * math.sqrt(self.T)
        return vega
