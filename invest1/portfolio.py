from __future__ import division
import QSTK.qstkutil.qsdateutil as du
#import QSTK.qstkutil.tsutil as tsu
import QSTK.qstkutil.DataAccess as da
import numpy as np

import datetime as dt
import matplotlib.pyplot as plt
import math
import itertools as it
#import pandas as pd


def chart(
    symbols=("AAPL", "GLD", "GOOG", "$SPX", "XOM", "msft"),
    start=dt.datetime(2005, 1, 1),
    end=dt.datetime(2014, 10, 31),  # data stops at 2013/1/1
    normalize=True,
    ):
    """Display a graph of the price history for the list of ticker symbols provided


    Arguments:
      symbols (list of str): Ticker symbols like "GOOG", "AAPL", etc
      start (datetime): The date at the start of the period being analyzed.
      end (datetime): The date at the end of the period being analyzed.
      normalize (bool): Whether to normalize prices to 1 at the start of the time series.
    """

    symbols = [s.upper() for s in symbols]
    timeofday = dt.timedelta(hours=16)
    timestamps = du.getNYSEdays(start, end, timeofday)

    c_dataobj = da.DataAccess('Yahoo')
    ls_keys = ['open', 'high', 'low', 'close', 'volume', 'actual_close']
    ldf_data = c_dataobj.get_data(timestamps, symbols, ls_keys)
    d_data = dict(zip(ls_keys, ldf_data))

    na_price = d_data['close'].values
    if normalize:
        na_price /= na_price[0, :]
    plt.clf()
    plt.plot(timestamps, na_price)
    plt.legend(symbols)
    plt.ylabel('Adjusted Close')
    plt.xlabel('Date')
    plt.savefig('chart.pdf', format='pdf')
    plt.grid(True)
    plt.show()
    return na_price


def portfolio_prices(
    symbols=("AAPL", "GLD", "GOOG", "$SPX", "XOM", "msft"),
    start=dt.datetime(2005, 1, 1),
    end=dt.datetime(2011, 12, 31),  # data stops at 2013/1/1
    normalize=True,
    allocation=None, 
    ):
    """Calculate the Sharpe Ratio and other performance metrics for a portfolio

    Arguments:
      symbols (list of str): Ticker symbols like "GOOG", "AAPL", etc
      start (datetime): The date at the start of the period being analyzed.
      end (datetime): The date at the end of the period being analyzed.
      normalize (bool): Whether to normalize prices to 1 at the start of the time series.
      allocation (list of float): The portion of the portfolio allocated to each equity.
    """    
    if allocation is None:
        allocation = [1. / len(symbols)] * len(symbols)
    if len(allocation) < len(symbols):
        allocation = list(allocation) + [1. / len(symbols)] * (len(symbols) - len(allocation))
    total = sum(allocation)
    allocation = np.array([(float(a) / total) for a in allocation])

    symbols = [s.upper() for s in symbols]
    timeofday = dt.timedelta(hours=16)
    timestamps = du.getNYSEdays(start, end, timeofday)

    c_dataobj = da.DataAccess('Yahoo')
    ls_keys = ['open', 'high', 'low', 'close', 'volume', 'actual_close']
    ldf_data = c_dataobj.get_data(timestamps, symbols, ls_keys)
    d_data = dict(zip(ls_keys, ldf_data))

    na_price = d_data['close'].values
    if normalize:
        na_price /= na_price[0, :]
    na_price *= allocation
    return np.sum(na_price, axis=1)


def metrics(prices, fudge=True):
    """Calculate the volatiliy, average daily return, Sharpe ratio, and cumulative return

    Examples:
      >>> metrics(np.array([1,2,3,4]))
      >>> metrics(portfolio_prices(symbols=['AAPL', 'GLD', 'GOOG', 'XOM'], start=dt.datetime(2011,1,1), end=dt.datetime(2011,12,31), allocations=[0.4, 0.4, 0.0, 0.2]))
      0.0101467067654, 0.000657261102001, 1.02828403099, 1.16487261965
    """
    prices = np.array([float(p) for p in prices])
    if isinstance(fudge, (float, int)):
        fudge = float(fudge)
    elif fudge == True:
        fudge = (len(prices) - 1.) / len(prices)
    else:
        fudge = 1.
    daily_returns = np.diff(prices) / prices[0:-1]
    mean = fudge * np.average(daily_returns)
    variance = fudge * np.sum((daily_returns - mean) * (daily_returns - mean)) / len(daily_returns)
    return math.sqrt(variance), mean, mean * np.sqrt(252.) / np.sqrt(variance), (prices[-1] - prices[0]) / prices[0] + 1.


def prices(symbol='$DJI', start=dt.datetime(2009,2,1), end=dt.datetime(2012,7,31)):
    symbol = symbol.upper()
    timeofday = dt.timedelta(hours=16)
    timestamps = du.getNYSEdays(start, end, timeofday)

    c_dataobj = da.DataAccess('Yahoo')
    ls_keys = ['open', 'high', 'low', 'close', 'volume', 'actual_close']
    ldf_data = c_dataobj.get_data(timestamps, [symbol], ls_keys)
    d_data = dict(zip(ls_keys, ldf_data))
    na_price = d_data['close'].values
    return na_price[:,0]


def simulate(symbols=("AAPL", "GLD", "GOOG", "$SPX", "XOM", "msft"),
    start=dt.datetime(2005, 1, 1),
    end=dt.datetime(2011, 12, 31),  # data stops at 2013/1/1
    normalize=True,
    allocation=None,
    fudge=True,
    ):
    p = portfolio_prices(symbols=symbols, start=start, end=end, normalize=normalize, allocation=allocation)
    return metrics(p, fudge=fudge)

def optimize_allocation(symbols=("AAPL", "GLD", "GOOG", "$SPX", "XOM"),
                        start=dt.datetime(2005, 1, 1),
                        end=dt.datetime(2011, 12, 31),  
                        normalize=True,
                        ):
    N = len(symbols)
    alloc = it.product(range(11), repeat=N-1)
    best_results = [0, 0, 0, 0]

    for a in alloc:
        if sum(a) > 10:
            continue
        last_alloc = 10 - sum(a)
        allocation = 0.1 * np.array(list(a) + [last_alloc])

        results = simulate(symbols=symbols, start=start, end=end, normalize=normalize, 
                           allocation=allocation)
        if results[2] > best_results[2]:
            best_results = results
            best_allocation = allocation
            print allocation
            print results
    return best_results, best_allocation
