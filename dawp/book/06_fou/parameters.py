#
# Model Parameters for European Call Option
# in Binomial Model
# 06_fou/parameters.py
#
# (c) Dr. Yves J. Hilpisch
# Derivatives Analytics with Python
#
from math import exp, sqrt

# Model and Option Parameters
S0 = 100.0  # index level
K = 100.0  # option strike
T = 1.0  # maturity date
r = 0.05  # risk-less short rate
sigma = 0.2  # volatility

def get_binomial_parameters(M=100):
    # Time Parameters
    dt = T / M  # length of time interval
    df = exp(-r * dt)  # discount per interval

    # Binomial Parameters
    u = exp(sigma * sqrt(dt))  # up movement
    d = 1 / u  # down movement
    q = (exp(r * dt) - d) / (u - d)  # martingale branch probability
    return dt, df, u, d, q
