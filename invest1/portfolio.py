#!/usr/bin/env python
from __future__ import division, unicode_literals
"""Portfolio evaluation and optimization utilities"""

import os
import csv
import math
import itertools
import datetime

import numpy as np
import matplotlib.pyplot as plt

import QSTK.qstkutil.qsdateutil as du
#import QSTK.qstkutil.tsutil as tsu
import QSTK.qstkutil.DataAccess as da

#import pandas as pd


def chart(
    symbols=("AAPL", "GLD", "GOOG", "$SPX", "XOM", "msft"),
    start=datetime.datetime(2005, 1, 1),
    end=datetime.datetime(2014, 10, 31),  # data stops at 2013/1/1
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
    timeofday = datetime.timedelta(hours=16)
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
    start=datetime.datetime(2005, 1, 1),
    end=datetime.datetime(2011, 12, 31),  # data stops at 2013/1/1
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
    timeofday = datetime.timedelta(hours=16)
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
      >>> metrics(np.array([1,2,3,4])) == {'mean': 0.61111111111111105, 'return': 4.0, 'sharpe': 34.245718429742873, 'std': 0.28327886186626583}
      True
      >>> metrics(portfolio_prices(symbols=['AAPL', 'GLD', 'GOOG', 'XOM'], start=datetime.datetime(2011,1,1), end=datetime.datetime(2011,12,31), allocations=[0.4, 0.4, 0.0, 0.2])
      ...        ) == {'std': 0.0101467067654, 'mean': 0.000657261102001, 'sharpe': 1.02828403099, 'return': 1.16487261965} 
      True
    """
    if isinstance(prices, basestring) and os.path.isfile(prices):
        values = []
        with csv.reader(open(prices), dialect='excel', quoting=csv.QUOTE_MINIMAL) as reader:
            for row in reader:
                values += [row[-1]]
        prices = values
    p.metrics(values)

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
    return {'std': math.sqrt(variance), 'mean': mean, 
            'sharpe': mean * np.sqrt(252.) / np.sqrt(variance), 
            'return': (prices[-1] - prices[0]) / prices[0] + 1.}


def prices(symbol='$DJI', start=datetime.datetime(2009,2,1), end=datetime.datetime(2012,7,31)):
    symbol = symbol.upper()
    timeofday = datetime.timedelta(hours=16)
    timestamps = du.getNYSEdays(start, end, timeofday)

    c_dataobj = da.DataAccess('Yahoo')
    ls_keys = ['open', 'high', 'low', 'close', 'volume', 'actual_close']
    ldf_data = c_dataobj.get_data(timestamps, [symbol], ls_keys)
    d_data = dict(zip(ls_keys, ldf_data))
    na_price = d_data['close'].values
    return na_price[:,0]


def simulate(symbols=("AAPL", "GLD", "GOOG", "$SPX", "XOM", "msft"),
    start=datetime.datetime(2005, 1, 1),
    end=datetime.datetime(2011, 12, 31),  # data stops at 2013/1/1
    normalize=True,
    allocation=None,
    fudge=True,
    ):
    p = portfolio_prices(symbols=symbols, start=start, end=end, normalize=normalize, allocation=allocation)
    return metrics(p, fudge=fudge)

def optimize_allocation(symbols=("AAPL", "GLD", "GOOG", "$SPX", "XOM"),
                        start=datetime.datetime(2005, 1, 1),
                        end=datetime.datetime(2011, 12, 31),  
                        normalize=True,
                        ):
    N = len(symbols)
    alloc = itertools.product(range(11), repeat=N-1)
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
