#
# Circular convolution of two 1-dim vectors
# 06_fou/convolution.py
#
# (c) Dr. Yves J. Hilpisch
# Derivatives Analytics with Python
#
import numpy as np

#
# Function Definitions
#


def revpy(a):
    ''' Reversing the order of the vector's numbers (for loop). '''
    a = np.array(a)
    n = len(a)
    c = np.zeros(n, dtype=np.float)
    c[0] = a[0]
    for j in range(1, n):
        c[j] = a[n - j]
    return c


def revnp(a):
    ''' Reversing the order of the vector's numbers (NumPy version). '''
    b = a.copy()
    b[1:] = b[1:][::-1]
    return b


def convolution(a, b):
    ''' Convolution of two vectors. '''
    if len(a) != len(b):
        raise ValueError("Lengths of vectors do not match.")
    n = len(a)
    c = np.zeros(n, dtype=np.float)
    for j in range(n):
        s = 0
        for k in range(n):
            if j - k >= 0:
                s += a[j - k] * b[k]
            else:
                s += a[j - k + n] * b[k]
        c[j] = s
    return c
