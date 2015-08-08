#
# Model Parameters for European Call Option
# and Binomial Model
# A_pyt/c_parameters.py
#
import math

# Option Parameters
S0 = 105.0  # initial index level
K = 100.0  # strike price
T = 1.  # call option maturity
r = 0.05  # constant short rate
vola = 0.25  # constant volatility factor of diffusion

# Time Parameters
M = 1000  # time steps
dt = T / M  # length of time interval
df = math.exp(-r * dt)  # discount factor per time interval

# Binomial Parameters
u = math.exp(vola * math.sqrt(dt))  # up-movement
d = 1 / u  # down-movement
q = (math.exp(r * dt) - d) / (u - d)  # martingale probability
