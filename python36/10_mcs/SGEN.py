for t in range(1, M + 1, 1):
    ran = np.dot(CM, rand[:, t])
    if momatch:
        bias = np.mean(np.sqrt(v[t]) * ran[row] * sdt)
    if s_disc == 'Log':
        S[t] = S[t - 1] * np.exp((r - 0.5 * v[t]) * dt +
                                 np.sqrt(v[t]) * ran[row] * sdt - bias)
    elif s_disc == 'Naive':
        S[t] = S[t - 1] * (math.exp(r * dt) +
                           np.sqrt(v[t]) * ran[row] * sdt - bias)
