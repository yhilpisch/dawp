#
# Cubic Spline Interpolation
# A_pyt/i_SPLINE.py
#
import numpy as np
import scipy.interpolate as sci
import matplotlib.pyplot as plt

# Interpolation
x = np.linspace(0.0, np.pi / 2, 20)  # x values
y = np.cos(x)  # function values to interpolate
gp = sci.splrep(x, y, k=3)  # cubic spline interpolatiln
gy = sci.splev(x, gp, der=0)  # calculate interpolated values

# Graphical Output
plt.figure()
plt.plot(x, y, 'b', label='cosine')  # plot original function values
plt.plot(x, gy, 'ro', label='cubic splines') 
  # plot interpolated function values
plt.legend(loc=0)
plt.grid(True)
