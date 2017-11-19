#
# Numerically Integrate a Function
# A_pyt/k_INT.py
#
import numpy as np
from scipy.integrate import quad

# Numerical Integration


def f(x):
    ''' Function to Integrate. '''
    return np.exp(x)


int_value = quad(lambda u: f(u), 0, 1)[0]

# Output
print("Value of the integral is %10.9f" % int_value)
