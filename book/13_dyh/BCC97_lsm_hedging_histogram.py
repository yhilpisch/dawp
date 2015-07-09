#
# Delta Hedging an American Put Option in BCC97
# via Least Squares Monte Carlo (Multiple Replications)
# 13_dyh/BCC97_lsm_hedging_histogram.py
#
# (c) Dr. Yves s. Hilpisch
# Derivatives Analytics with Python
#
from BCC97_lsm_hedging_algorithm import *
from CIR_zcb_valuation_gen import B
from BSM_lsm_hedging_histogram import plot_hedge_histogram

#
# Simulation
#
T = 1.0
a = 1.0  # a from the interval [0.0, 2.0]
dis = 0.05  # change of S[t] in percent to estimate derivative
dt = T / M
np.random.seed(50000)

def BCC97_hedge_simulation(M=50, I=10000):
    ''' Monte Carlo simualtion of dynamic hedging paths
    for American put option in BSM model. '''
    #
    # Initializations
    #
    po = np.zeros(M + 1, dtype=np.float)  # vector for portfolio values
    delt = np.zeros(M + 1, dtype=np.float)  # vector for deltas
    ds = dis * S0
    V_1, S, r, v, ex, rg, h, dt = BCC97_lsm_put_value(S0 + (2 - a) * ds,
                                                      K, T, M, I)
    # 'data basis' for delta hedging
    V_2 = BCC97_lsm_put_value(S0 - a * ds, K, T, M, I)[0]
    delt[0] = (V_1 - V_2) / (2 * ds)
    V0LSM = BCC97_lsm_put_value(S0, K, T, M, I)[0]  # initial option value for S0
    po[0] = V0LSM  # initial portfolio values

    #
    # Hedge Runs
    #
    pl_list = []
    runs = min(I, 10000)
    for run in range(runs):
        bo = V0LSM - delt[0] * S0  # initial bond position value
        p = run
        run += 1
        for t in range(1, M + 1, 1):
            if ex[t, p] == 0:
                df = math.exp((r[t, p] + r[t - 1, p]) / 2 * dt)
                if t != M:
                    po[t] = delt[t - 1] * S[t, p] + bo * df  # portfolio payoff
                    ds = dis * S[t, p]
                    sd = S[t, p] + (2 - a) * ds  # disturbed index level
                    stateV_A = [sd * v[t, p] * r[t, p],
                                sd * v[t, p],
                                sd * r[t, p],
                                v[t, p] * r[t, p],
                                sd ** 2,
                                v[t, p] ** 2,
                                r[t, p] ** 2,
                                sd,
                                v[t, p],
                                r[t, p],
                                1]
                        # state vector for S[t, p] + (2.0 - a) * ds
                    stateV_A.reverse()
                    V0A = max(0, np.dot(rg[t], stateV_A))
                    # revaluation via regression
                    sd = S[t, p] - a * ds  # disturbed index level
                    stateV_B = [sd * v[t, p] * r[t, p],
                                sd * v[t, p],
                                sd * r[t, p],
                                v[t, p] * r[t, p],
                                sd ** 2,
                                v[t, p] ** 2,
                                r[t, p] ** 2,
                                sd,
                                v[t, p],
                                r[t, p],
                                1]
                        # state vector for S[t, p] - a * ds
                    stateV_B.reverse()
                    V0B = max(0, np.dot(rg[t], stateV_B))
                    # revaluation via regression
                    delt[t] = (V0A - V0B) / (2 * ds)
                else:
                    po[t] = delt[t - 1] * S[t, p] + bo * df
                    delt[t] = 0.0
                bo = po[t] - delt[t] * S[t, p]
            else:
                po[t] = delt[t - 1] * S[t, p] + bo * df
                break
        alpha_t = [kappa_r, theta_r, sigma_r, r0, 0.0, t * dt]
        pl = (po[t] - h[t, p]) * B(alpha_t)
        if run % 1000 == 0:
            print "run %5d   p/l %8.3f" % (run, pl)
        pl_list.append(pl)
    pl_list = np.array(pl_list)

    #
    # Results Output
    #
    print "\nSUMMARY STATISTICS FOR P&L"
    print "---------------------------------"
    print "Dynamic Replications %12d" % runs
    print "Time Steps           %12d" % M
    print "Paths for Valuation  %12d" % I
    print "Maximum              %12.3f" % max(pl_list)
    print "Average              %12.3f" % np.mean(pl_list)
    print "Median               %12.3f" % np.median(pl_list)
    print "Minimum              %12.3f" % min(pl_list)
    print "---------------------------------"

    return pl_list