#
# Loop Version --- Iterations in Python
#
# Array Initialization for Inner Values
iv = np.zeros((M + 1, M + 1), dtype=np.float)
z = 0
for j in xrange(0, M + 1, 1):
    for i in xrange(z + 1):
        iv[i, j] = max(S[i, j] - K, 0)
    z += 1

#
# Vectorized Version --- Iterations on NumPy Level
#
# Array Initialization for Inner Values
pv = maximum(S - K, 0)
