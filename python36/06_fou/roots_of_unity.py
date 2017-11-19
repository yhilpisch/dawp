#
# Plotting Spokes and Points on a Circle
# with Complex Numbers
# 06_fou/roots_of_unity.py
#
# (c) Dr. Yves J. Hilpisch
# Derivatives Analytics with Python
#
import numpy as np
import matplotlib.pyplot as plt


def generate_subplot(n):
    y = np.exp(1j * 2 * np.pi / n) ** np.arange(1, n + 1)
    for l in range(n):
        plt.plot(y[l].real, y[l].imag, 'ro')
        plt.plot((0, y[l].real), (0.0, y[l].imag), 'b')
    plt.axis([-1.1, 1.1, -1.1, 1.1])
    plt.xlabel('$n=%s$' % n)


def generate_plot():
    plt.figure(figsize=(10, 7))
    # first sub-plot for n=5
    plt.subplot(121)
    generate_subplot(n=5)

    # second sub-plot for n=30
    plt.subplot(122)
    generate_subplot(n=30)

    plt.subplots_adjust(left=0.05, bottom=0.2, top=0.8, right=1.0)
