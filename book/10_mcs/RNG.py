def random_number_generator(M, I):
    if antipath:
        rand = np.random.standard_normal((2, M + 1, I / 2))
        rand = np.concatenate((rand, -rand), 2)
    else:
        rand = np.random.standard_normal((2, M + 1, I))
    if momatch:
        rand = rand / np.std(rand)
        rand = rand - np.mean(rand)
    return rand
