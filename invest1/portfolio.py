#!/usr/bin/env python
from __future__ import division, unicode_literals
"""Portfolio evaluation and optimization utilities"""

import os
import csv
import math
import itertools
import datetime

from dateutil.parser import parse as parse_date
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import QSTK.qstkutil.qsdateutil as du
#import QSTK.qstkutil.tsutil as tsu
import QSTK.qstkutil.DataAccess
da = QSTK.qstkutil.DataAccess.DataAccess('Yahoo')
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

    ls_keys = ['open', 'high', 'low', 'close', 'volume', 'actual_close']
    ldf_data = da.get_data(timestamps, symbols, ls_keys)
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


def chart_series(series, market_sym='$SPX', price='close', normalize=True):
    """Display a graph of the price history for the list of ticker symbols provided


    Arguments:
      series (dataframe, list of str, or list of tuples): 
        datafram (Timestamp or Datetime for index)
          other columns are float y-axis values to be plotted
        list of str: 1st 3 comma or slash-separated integers are the year, month, day
          others are float y-axis values
        list of tuples: 1st 3 integers are year, month, day 
          otehrs are float y-axis values
      market_sym (str): ticker symbol of equity or comodity to plot along side the series
      price (str): which market data value ('close', 'actual_close', 'volume', etc) to use 
         for the market symbol for comparison to the series 
      normalize (bool): Whether to normalize prices to 1 at the start of the time series.
    """
    series = make_dataframe(series)
    start = series.index[0] or datetime.datetime(2008, 1, 1)
    end = series.index[-1] or datetime.datetime(2009, 12, 28)
 
    timeofday = datetime.timedelta(hours=16)
    timestamps = du.getNYSEdays(start, end, timeofday)
    if market_sym:
        if isinstance(market_sym, basestring):
            market_sym = [market_sym.upper().strip()]
        reference_prices = da.get_data(timestamps, market_sym, [price])
        reference_dict = dict(zip(market_sym, reference_prices))
        for sym in market_sym:
            series[sym] = pd.Series(reference_dict[sym], index=timestamps)
    # na_price = reference_dict[price].values
    # if normalize:
    #     na_price /= na_price[0, :]
    series.plot()
    # plt.clf()
    # plt.plot(timestamps, na_price)
    # plt.legend(symbols)
    # plt.ylabel(price.title())
    # plt.xlabel('Date')
    # # plt.savefig('portfolio.chart_series.pdf', format='pdf')
    plt.grid(True)
    plt.show()
    return 


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

    ls_keys = ['open', 'high', 'low', 'close', 'volume', 'actual_close']
    ldf_data = da.get_data(timestamps, symbols, ls_keys)
    d_data = dict(zip(ls_keys, ldf_data))

    na_price = d_data['close'].values
    if normalize:
        na_price /= na_price[0, :]
    na_price *= allocation
    return np.sum(na_price, axis=1)

COLUMN_SEP = re.compile(r'[,/;]')

def make_dataframe(prices, num_prices=None):
    """Convert a file, list of strings, or list of tuples into a Pandas DataFrame

    Arguments:
      num_prices (int): if not null, the number of columns (from right) that contain numeric values
    """
    if isinstance(prices, basestring) and os.path.isfile(prices):
        prices = open(prices, 'rU')
    if isinstance(prices, file):
        values = []
        csvreader = csv.reader(prices, dialect='excel', quoting=csv.QUOTE_MINIMAL)
        for row in csvreader:
            print row
            values += [row[-1]]
        prices.close()
        prices = values
    if isinstance(prices[0], basestring):
        prices = [COLUMN_SEP.split(row) for row in prices]
    index = []
    if isinstance(prices[0][0], (datetime.date, datetime.datetime, datetime.time)):
        index = [prices[0] for row in prices]
        for i, row in prices:
            prices[i] = row[1:]
    # try to convert all strings to something numerical:
    elif all(all(isinstance(value, basestring) for value in row) for row in prices):
        for i, row in enumerate(prices):
            for j, value in enumerate(row):
                try:
                    prices[i][j] = int(prices[i][j])
                except:
                    try:
                        prices[i][j] = float(prices[i][j])
                    except:
                        try:
                            # this is a probably a bit too forceful
                            prices[i][j] = parse_date(prices[i][j])
                        except:
                            pass
    if not index and isinstance(prices[0], (tuple, list)) and len(prices[0]) > 3:
        for i, row in enumerate(prices):
            try:
                index += [datetime.datetime(*row[:3])]
                prices[i] = row[3:]
            except:
                break
    # TODO: label the columns somehow (if first row is a bunch of strings/header)
    if len(index) == len(prices):
        df = DataFrame(prices, index=index)
    else:
        df = DataFrame(prices)


def metrics(prices, fudge=False, sharpe_days=252):
    """Calculate the volatiliy, average daily return, Sharpe ratio, and cumulative return

    Arguments:
      prices (file or basestring or iterable): path to file or file pointer or sequence of prices/values of a portfolio or equity
      fudge (bool): Whether to use Tucker Balche's erroneous division by N or the more accurate N-1 for stddev of returns
      sharpe_days: Number of trading days in a year. Sharpe ratio = sqrt(sharpe_days) * total_return / std_dev_of_daily_returns

    Examples:
      >>> metrics(np.array([1,2,3,4])) == {'mean': 0.61111111111111105, 'return': 4.0, 'sharpe': 34.245718429742873, 'std': 0.28327886186626583}
      True
      >>> metrics(portfolio_prices(symbols=['AAPL', 'GLD', 'GOOG', 'XOM'], start=datetime.datetime(2011,1,1), end=datetime.datetime(2011,12,31), allocations=[0.4, 0.4, 0.0, 0.2])
      ...        ) == {'std': 0.0101467067654, 'mean': 0.000657261102001, 'sharpe': 1.02828403099, 'return': 1.16487261965} 
      True
    """
    if isinstance(prices, basestring) and os.path.isfile(prices):
        prices = open(prices, 'rU')
    if isinstance(prices, file):
        values = []
        csvreader = csv.reader(prices, dialect='excel', quoting=csv.QUOTE_MINIMAL)
        for row in csvreader:
            print row
            values += [row[-1]]
        prices.close()
        prices = values
    if isinstance(prices[0], (tuple, list)):
        prices = [row[-1] for row in prices]
    if sharpe_days == None:
        sharpe_days = len(prices)
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
            'sharpe': mean * np.sqrt(sharpe_days) / np.sqrt(variance), 
            'return': (prices[-1] - prices[0]) / prices[0] + 1.}


def prices(symbol='$DJI', start=datetime.datetime(2009,2,1), end=datetime.datetime(2012,7,31)):
    symbol = symbol.upper()
    timeofday = datetime.timedelta(hours=16)
    timestamps = du.getNYSEdays(start, end, timeofday)

    ls_keys = ['open', 'high', 'low', 'close', 'volume', 'actual_close']
    ldf_data = da.get_data(timestamps, [symbol], ls_keys)
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
