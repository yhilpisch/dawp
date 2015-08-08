#
# Dynamic Hedging of American Put Option in BSM Model
# with Least Squares Monte Carlo -- Histogram
# 13_dyh/BSM_lsm_hedging_histogram.py
#
# (c) Dr. Yves s. Hilpisch
# Derivatives Analytics with Python
#
from BSM_lsm_hedging_algorithm import *

def BSM_dynamic_hedge_mcs(M=50, I=10000):
    ''' Monte Carlo simualtion of dynamic hedging paths
    for American put option in BSM model. '''
    #
    # Initial Delta
    #
    ds = 0.01
    V_1, S, ex, rg, h, dt = BSM_lsm_put_value(S0 + ds, M, I)
    V_2 = BSM_lsm_put_value(S0, M, I)[0]
    del_0 = (V_1 - V_2) / ds

    print"Value of American Put Option is %8.3f" % V_2
    print"Delta t=0 is                    %8.3f" % del_0
    #
    # Dynamic Hedging Runs
    #
    pl_list = []
    run = 0
    runs = min(I, 10000)
    for run in xrange(runs):
        p = run
        run += 1
        delta = np.zeros(M + 1, dtype=np.float)  # vector for deltas
        for t in xrange(0, M, 1):
            if ex[t - 1, p] == 0:  # if option is alive
                St = S[t, p]  # relevant index level
                diff = (np.polyval(rg[t, :], St + ds)
                      - np.polyval(rg[t, :], St))
                         # numerator of difference quotient
                delta[t] = diff / ds  # delta as difference quotient
            else:
                break
        delta[0] = del_0
        po = np.zeros(t, dtype=np.float)  # vector for portfolio values
        vt = np.zeros(t, dtype=np.float)  # vector for option values
        vt[0] = V_2  # initial option value
        po[0] = V_2  # initial portfolio value
        bo = V_2 - delta[0] * S0  # initial bond position value
        for s in range(1, t, 1):  # for all times up to i-1
            po[s] = delta[s - 1] * S[s, p] + bo * math.exp(r * dt) 
              # portfolio payoff
            bo = po[s] - delta[s] * S[s, p]  # bond position value
            if s == t - 1:  # at exercise/expiration date
                vt[s] = h[s, p]  # option value equals inner value
                pl = (po[s] - vt[s]) * math.exp(-r * t * dt)
                  # discounted difference between option and portfolio value
                if run % 1000 == 0:
                    print "run %5d   p/l %8.3f" % (run, pl)
                pl_list.append(pl)  # collect all differences
    pl_list = np.array(pl_list)

    #
    # Summary Results Output
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

def plot_hedge_histogram(pl_list):
    ''' Plot of P/L histogram. '''
    #
    # Graphical Output
    #
    plt.figure(figsize=(8, 6))
    plt.grid()
    plt.hist(pl_list, 75)
    plt.xlabel('profit/loss')
    plt.ylabel('frequency')