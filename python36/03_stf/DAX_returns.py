#
# Analyzing DAX Index Quotes and Returns
# Source: http://finance.yahoo.com
# 03_stf/DAX_returns.py
#
# (c) Dr. Yves J. Hilpisch
# Derivatives Analytics with Python
#
from GBM_returns import *

# Read Data for DAX from the Web


def read_dax_data():
    ''' Reads historical DAX data from Yahoo! Finance, calculates log returns,
    realized variance and volatility.'''
    DAX = pd.read_csv('http://hilpisch.com/tr_eikon_eod_data_long.csv',
                      index_col=0, parse_dates=True)['.GDAXI']
    DAX = pd.DataFrame(DAX)
    DAX = DAX.loc['2004-09-30':'2014-09-30']
    DAX.rename(columns={'.GDAXI': 'index'}, inplace=True)
    DAX['returns'] = np.log(DAX['index'] / DAX['index'].shift(1))
    DAX['rea_var'] = 252 * np.cumsum(DAX['returns'] ** 2) / np.arange(len(DAX))
    DAX['rea_vol'] = np.sqrt(DAX['rea_var'])
    DAX = DAX.dropna()
    return DAX


def count_jumps(data, value):
    ''' Counts the number of return jumps as defined in size by value. '''
    jumps = np.sum(np.abs(data['returns']) > value)
    return jumps
