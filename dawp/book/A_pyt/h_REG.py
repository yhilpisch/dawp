#
# Ordinary Least Squares Regression
# A_pyt/h_REG.py
#
import numpy as np
import matplotlib.pyplot as plt

# Regression
x = np.linspace(0.0, np.pi / 2, 20)  # x values
y = np.cos(x)  # y values, i.e. those values to regress
g1 = np.polyfit(x, y, 0)  # OLS of degree 1
g2 = np.polyfit(x, y, 1)  # OLS of degree 2
g3 = np.polyfit(x, y, 2)  # OLS of degree 3

g1y = np.polyval(g1, x)  # calculate regressed values for x vector
g2y = np.polyval(g2, x)
g3y = np.polyval(g3, x)

# Graphical Output
plt.figure()  # intialize new figure
plt.plot(x, y, 'r', lw=3, label='cosine')  # plot original function values
plt.plot(x, g1y, 'mx', label='constant')  # plot regression function values
plt.plot(x, g2y, 'bo', label='linear')
plt.plot(x, g3y, 'g>', label='quadratic')
plt.legend(loc=0)
plt.grid(True)
