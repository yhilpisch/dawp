#
# Dynamic Hedging of American Put Option in BSM Model
# with Least Squares Monte Carlo
# 13_dyh/BSM_lsm_hedging_algorithm.py
#
# (c) Dr. Yves J. Hilpisch
# Derivatives Analytics with Python
#
import math
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams['font.family'] = 'serif'
#
# Parameters
#
S0 = 36.0  # initial stock value
K = 40.0  # strike price
T = 1.0  # time to maturity
r = 0.06  # risk-less short rate
sigma = 0.20  # volatility of stock value
M = 50  # number of time steps
I = 50000  # number of paths


#
# Valuation
#
D = 9  # number of regression functions


def BSM_lsm_put_value(S0, M, I):
    ''' Function to value an American put option by LSM algorithm.

    Parameters
    ==========
    S0: float
        initial index level
    M: int
        number of time steps

    Returns
    =======
    V0: float
        LSM Monte Carlo estimator of American put option value
    S: NumPy array
        simulated index level paths
    ex: NumPy array
        exercise matrix
    rg: NumPy array
        regression coefficients
    h: NumPy array
        inner value matrix
    dt: float
        length of time interval
    '''
    rand = np.random.standard_normal((M + 1, I))  # random numbers
    dt = T / M   # length of time interval
    df = math.exp(-r * dt)  # discount factor
    S = np.zeros((M + 1, I), dtype=np.float)  # stock price matrix
    S[0] = S0  # initial values
    for t in range(1, M + 1, 1):  # stock price at t
        S[t] = S[t - 1] * (np.exp((r - sigma ** 2 / 2) * dt +
                                  sigma * math.sqrt(dt) * rand[t]))
    h = np.maximum(K - S, 0)  # inner values
    V = np.maximum(K - S, 0)  # value matrix
    ex = np.zeros((M + 1, I), dtype=np.float)   # exercise matrix
    C = np.zeros((M + 1, I), dtype=np.float)   # continuation value matrix
    rg = np.zeros((M + 1, D + 1), dtype=np.float)
    # matrix for reg. coefficients
    for t in range(M - 1, 0, -1):
        rg[t] = np.polyfit(S[t], V[t + 1] * df, D)
        # regression in step i
        C[t] = np.polyval(rg[t], S[t])
        # estimated continuation values
        C[t] = np.where(C[t] < 0, 0., C[t])
        # correction for neg C
        V[t] = np.where(h[t] >= C[t],
                        h[t], V[t + 1] * df)  # exercise decision
        ex[t] = np.where(h[t] >= C[t], 1, 0)
        # exercise decision (yes=1)
    V0 = np.sum(V[1]) / I * df
    return V0, S, ex, rg, h, dt


def BSM_hedge_run(p=0):
    ''' Implements delta hedging for a single path. '''
    np.random.seed(50000)
    #
    # Initial Delta
    #
    ds = 0.01
    V_1, S, ex, rg, h, dt = BSM_lsm_put_value(S0 + ds, M, I)
    V_2 = BSM_lsm_put_value(S0, M, I)[0]
    del_0 = (V_1 - V_2) / ds

    #
    # Dynamic Hedging
    #
    delt = np.zeros(M + 1, dtype=np.float)  # vector for deltas
    print
    print("APPROXIMATION OF FIRST ORDER ")
    print("-----------------------------")
    print(" %7s | %7s | %7s " % ('step', 'S_t', 'Delta'))
    for t in range(1, M, 1):
        if ex[t, p] == 0:  # if option is alive
            St = S[t, p]  # relevant index level
            diff = (np.polyval(rg[t], St + ds) -
                    np.polyval(rg[t], St))
            # numerator of difference quotient
            delt[t] = diff / ds  # delta as difference quotient
            print(" %7d | %7.2f | %7.2f" % (t, St, delt[t]))
            if (S[t, p] - S[t - 1, p]) * (delt[t] - delt[t - 1]) < 0:
                print("          wrong")
        else:
            break

    delt[0] = del_0
    print()
    print("DYNAMIC HEDGING OF AMERICAN PUT (BSM)")
    print("---------------------------------------")
    po = np.zeros(t, dtype=np.float)  # vector for portfolio values
    vt = np.zeros(t, dtype=np.float)  # vector for option values
    vt[0] = V_1
    po[0] = V_1
    bo = V_1 - delt[0] * S0  # bond position value
    print("Initial Hedge")
    print("Stocks             %8.3f" % delt[0])
    print("Bonds              %8.3f" % bo)
    print("Cost               %8.3f" % (delt[0] * S0 + bo))

    print()
    print("Regular Rehedges ")
    print(68 * "-")
    print("step|" + 7 * " %7s|" % ('S_t', 'Port', 'Put',
                                   'Diff', 'Stock', 'Bond', 'Cost'))
    for j in range(1, t, 1):
        vt[j] = BSM_lsm_put_value(S[j, p], M - j, I)[0]
        po[j] = delt[j - 1] * S[j, p] + bo * math.exp(r * dt)
        bo = po[j] - delt[j] * S[j, p]  # bond position value
        print("%4d|" % j + 7 * " %7.3f|" % (S[j, p], po[j], vt[j],
                                            (po[j] - vt[j]), delt[j],
                                            bo, delt[j] * S[j, p] + bo))

    errs = po - vt  # hedge errors
    print("MSE             %7.3f" % (np.sum(errs ** 2) / len(errs)))
    print("Average Error   %7.3f" % (np.sum(errs) / len(errs)))
    print("Total P&L       %7.3f" % np.sum(errs))
    return S[:, p], po, vt, errs, t


def plot_hedge_path(S, po, vt, errs, t):
    #
    # Graphical Output
    #
    tl = np.arange(t)
    plt.figure(figsize=(8, 6))
    plt.subplot(311)
    plt.plot(tl, S[tl], 'r')
    plt.ylabel('index level')
    plt.subplot(312)
    plt.grid(True)
    plt.plot(tl, po[tl], 'r-.', label='portfolio value', lw=2)
    plt.plot(tl, vt[tl], 'b', label='option value', lw=1)
    plt.ylabel('value')
    plt.legend(loc=0)
    ax = plt.axis()
    plt.subplot(313)
    wi = 0.3
    diffs = po[tl] - vt[tl]
    plt.bar(tl - wi / 2, diffs, color='b', width=wi)
    plt.ylabel('difference')
    plt.xlabel('time step')
    plt.axis([ax[0], ax[1], min(diffs) * 1.1, max(diffs) * 1.1])
    plt.tight_layout()
