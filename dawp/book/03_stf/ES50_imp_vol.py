#
# Black-Scholes-Merton Implied Volatilities of
# Call Options on the EURO STOXX 50
# Option Quotes from 30. September 2014
# Source: www.eurexchange.com, www.stoxx.com
# 03_stf/ES50_imp_vol.py
#
# (c) Dr. Yves J. Hilpisch
# Derivatives Analytics with Python
#
import numpy as np
import pandas as pd
from BSM_imp_vol import call_option
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['font.family'] = 'serif'

# Pricing Data
pdate = pd.Timestamp('30-09-2014')

#
# EURO STOXX 50 index data
#

# URL of data file
es_url = 'http://www.stoxx.com/download/historical_values/hbrbcpe.txt'
# column names to be used
cols = ['Date', 'SX5P', 'SX5E', 'SXXP', 'SXXE',
        'SXXF', 'SXXA', 'DK5F', 'DKXF', 'DEL']
# reading the data with pandas
es = pd.read_csv(es_url,  # filename
                 header=None,  # ignore column names
                 index_col=0,  # index column (dates)
                 parse_dates=True,  # parse these dates
                 dayfirst=True,  # format of dates
                 skiprows=4,  # ignore these rows
                 sep=';',  # data separator
                 names=cols)  # use these column names
# deleting the helper column
del es['DEL']
S0 = es['SX5E']['30-09-2014']
r = -0.05

#
# Option Data
#
data = pd.HDFStore('./03_stf/es50_option_data.h5', 'r')['data']

#
# BSM Implied Volatilities
#

def calculate_imp_vols(data):
    ''' Calculate all implied volatilities for the European call options
    given the tolerance level for moneyness of the option.'''
    data['Imp_Vol'] = 0.0
    tol = 0.30  # tolerance for moneyness
    for row in data.index:
        t = data['Date'][row]
        T = data['Maturity'][row]
        ttm = (T - t).days / 365.
        forward = np.exp(r * ttm) * S0
        if (abs(data['Strike'][row] - forward) / forward) < tol:
            call = call_option(S0, data['Strike'][row], t, T, r, 0.2)
            data['Imp_Vol'][row] = call.imp_vol(data['Call'][row])
    return data

#
# Graphical Output
#
markers = ['.', 'o', '^', 'v', 'x', 'D', 'd', '>', '<']
def plot_imp_vols(data):
    ''' Plot the implied volatilites. '''
    maturities = sorted(set(data['Maturity']))
    plt.figure(figsize=(10, 5))
    for i, mat in enumerate(maturities):
        dat = data[(data['Maturity'] == mat) & (data['Imp_Vol'] > 0)]
        plt.plot(dat['Strike'].values, dat['Imp_Vol'].values,
                 'b%s' % markers[i], label=str(mat)[:10])
    plt.grid()
    plt.legend()
    plt.xlabel('strike')
    plt.ylabel('implied volatility')
