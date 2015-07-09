#
# First Program with Python
# A_pyt/a_first_program.py
#
import math

# Variable Definition
a = 3.0
b = 4

# Function Definition


def f(x):
    ''' Mathematical Function. '''
    return x ** 3 + x ** 2 - 2 + math.sin(x)

# Calculation
f_a = f(a)
f_b = f(b)

# Output
print "f(a) = %6.3f" % f_a
print "f(b) = %6.3f" % f_b
