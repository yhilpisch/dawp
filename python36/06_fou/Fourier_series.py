#
# Fourier Series for f(x) = abs(x) for -pi <= x <= pi
# 06_fou/Fourier_series.py
#
# (c) Dr. Yves J. Hilpisch
# Derivatives Analytics with Python
#
import numpy as np
import matplotlib.pyplot as plt

#
# Fourier series function
#


def fourier_series(x, n):
    ''' Generate Fourier Series from vector x for f(x) = abs(x)
    of order n.

    Parameters
    ==========
    x : float or array of floats
        input numbers
    n : int
        order of Fourier series

    Returns
    =======
    fourier_values : float or array of floats
        numbers according to Fourier series approximation
    '''
    fourier_values = np.pi / 2
    for i in range(1, n + 1):
        fourier_values += ((2 * ((-1) ** i - 1)) /
                           (np.pi * i ** 2) * np.cos(i * x))
    return fourier_values


def plot_fourier_series():
    # Data Generation
    x = np.linspace(-np.pi, np.pi, 100)
    y1 = fourier_series(x, 1)
    y2 = fourier_series(x, 5)

    # Data Plotting
    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.plot(x, abs(x), 'b', label='$f(x) = |x|$')
    plt.plot(x, y1, 'r-.', lw=3.0, label='Fourier series $n=1$')
    plt.legend(loc=9)
    plt.subplot(122)
    plt.plot(x, abs(x), 'b', label='$f(x) = |x|$')
    plt.plot(x, y2, 'r-.', lw=3.0, label='Fourier series $n=5$')
    plt.legend(loc=9)
