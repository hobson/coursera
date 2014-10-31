#!/usr/bin/env python
from __future__ import division, unicode_literals
"""Simulate a sequence of trades and portfolio performance over time

Examples:
  $ python sim.py trade 50000 orders.csv values.csv

  # input file
  $ cat orders.csv:
  2008, 12, 3, AAPL, BUY, 130
  2008, 12, 8, AAPL, SELL, 130
  2008, 12, 5, IBM, BUY, 50

  # output file
  $ cat values.csv
  2008, 12, 3, 50000.25
  2008, 12, 4, 50010.25
  2008, 12, 5, 50250.125
"""
import argparse
import sys
import csv
import datetime
import re
import math
import itertools
import os
import json
import copy

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from dateutil.parser import parse as parse_date

import QSTK.qstkutil.DataAccess as da
import QSTK.qstkutil.qsdateutil as du
import QSTK.qstkstudy.EventProfiler as ep

from pug import debug
from pug.decorators import memoize



#t = qstk.dateutil.getNYSEdays(datetime.datetime(2010,1,1), datetime.datetime(2010,2,1), datetime.timedelta(hours=16))
dataobj = da.DataAccess('Yahoo')

DATE_SEP = re.compile(r'[^0-9]')

def get_price(symbol='$SPY', date=(2010,1,1), price='actual_close'):
    # if hasattr(symbol, '__iter__'):
    #     return [get_price(sym, date=date, price=price) for sym in symbol]
    if isinstance(date, basestring):
        date = DATE_SEP.split(date)
    if isinstance(date, (tuple, list)):
        date = datetime.datetime(*[int(i) for i in date])
    if isinstance(date, datetime.date) or date.hour < 9 or date.hour > 16:
        date = datetime.datetime(date.year, date.month, date.day, 16)
    symbol = str(symbol).upper().strip()
    if symbol == '$CASH':
        return 1.0
    try:
        sym_price = dataobj.get_data([date], [symbol], [price])[0][symbol][0]
        print 'found {0} price of {1} on {2}'.format(symbol, sym_price, date)
        return sym_price
    except IndexError:
        raise
    except:
        print 'BAD DATE ({0}) or SYMBOL ({1})'.format(date, symbol) 
        return None


def portfolio_value(portfolio, date, price='close'):
    """Total value of a portfolio (dict mapping symbols to numbers of shares)

    $CASH used as symbol for USD
    """
    value = 0.0
    for (sym, sym_shares) in portfolio.iteritems():
        sym_price = None
        if sym_shares:
            sym_price = get_price(symbol=sym, date=date, price=price)
        # print sym, sym_shares, sym_price
        # print last_date, k, price
        if sym_price != None:
            if np.isnan(sym_price):
                print 'Invalid price, shares, value, total: ', sym_price, sym_shares, (float(sym_shares) * float(sym_price)) if sym_shares and sym_price else 'Invalid', value
                if sym_shares:
                    return float('nan')
            else:
                # print ('{0} shares of {1} = {2} * {3} = {4}'.format(sym_shares, sym, sym_shares, sym_price, sym_shares * sym_price))
                value += sym_shares * sym_price
                # print 'new price, value = {0}, {1}'.format(sym_price, value)
    return value


def trade(args):
    """Simulate a sequence of trades indicated in `infile` and write the portfolio value time series to `outfile`"""
    print args
    print vars(args)['funds']
    print args.funds
    portfolio = { '$CASH': args.funds }
    print portfolio
    csvreader = csv.reader(args.infile, dialect='excel', quoting=csv.QUOTE_MINIMAL)
    csvwriter = csv.writer(args.outfile, dialect='excel', quoting=csv.QUOTE_MINIMAL)
    detailed = not args.simple
    history = []
    portfolio_history = []

    #trading_days = du.getNYSEdays(datetime.datetime(2010,01,01), datetime.datetime(2012,01,01), datetime.timedelta(hours=16))
    for row in csvreader:
        print '-'*30 + ' CSV Row ' + '-'*30
        print ', '.join(row)
        trade_date = datetime.datetime(*[int(i) for i in (row[:3] + [16])])

        if history:
            last_date = datetime.datetime(*(history[-1][:3] + [16])) + datetime.timedelta(days=1)
            # print (date.date() - last_date).days
            while (trade_date - last_date).days > 0:
                print 'Filling in the blanks for {0}'.format(last_date)
                value = portfolio_value(portfolio, last_date, price='close')
                print '   the portfolio value on that date is: {0}'.format(value)
                assert(value != None)
                # NaN for porfolio value indicates a non-trading day
                if not np.isnan(value):
                    history += [[last_date.year, last_date.month, last_date.day] 
                                + (["$CASH", "0.0", "0.0"] if args.trades else [])
                                + ([json.dumps(portfolio)] if detailed else []) + [value]]
                    portfolio_history += [datetime.datetime(last_date.year, last_date.month, last_date.day, 16), portfolio]
                    csvwriter.writerow(history[-1])
                last_date += datetime.timedelta(days=1)
    
        trade_symbol = row[3]
        trade_shares = float(row[5])
        trade_sign = 1 - 2 * int(row[4].strip().upper()[0]=='S')
        # If this the first row in the CSV and the symbol is $CASH then it represents an initial deposit (Sell) or withdrawal (Buy) of cash
        # otherwise se need to add or deduct whatever security was bought or sold.
        # if not (trade_symbol == '$CASH') or history:
        portfolio[trade_symbol] = portfolio.get(trade_symbol, 0) + trade_sign * trade_shares
        trade_price = get_price(symbol=trade_symbol, date=trade_date, price='close')
        while trade_price == None or np.isnan(trade_price) or float(trade_price) == float('nan'):
            trade_date += datetime.timedelta(days=1)
            trade_price = get_price(symbol=trade_symbol, date=trade_date, price='close')
        #print trade_date, trade_symbol, trade_sign, trade_shares, trade_price
        if trade_price and trade_shares and trade_sign in (-1, 1):
            print 'spending cash: {0}'.format(trade_sign * trade_shares * trade_price)
            portfolio['$CASH'] = portfolio.get('$CASH',0.) - trade_sign * trade_shares * trade_price
        else:
            print 'ERROR: bad price, sign, shares: ', trade_price, trade_sign, trade_shares
        history += [[trade_date.year, trade_date.month, trade_date.day, trade_symbol, trade_sign, trade_shares] + ([json.dumps(portfolio)] if detailed else []) + [portfolio_value(portfolio, trade_date, price='close')]]
        csvwriter.writerow(history[-1])
    return metrics(history)

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


def chart_series(series, market_sym='$SPX', price='actual_close', normalize=True):
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

    timestamps = du.getNYSEdays(start, end, datetime.timedelta(hours=16))
    if market_sym:
        if isinstance(market_sym, basestring):
            market_sym = [market_sym.upper().strip()]
        reference_prices = da.get_data(timestamps, market_sym, [price])[0]
        reference_dict = dict(zip(market_sym, reference_prices))
        for sym, market_data in reference_dict.iteritems():
            series[sym] = pd.Series(market_data, index=timestamps)
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
    return series


def portfolio_prices(
    symbols=("AAPL", "GLD", "GOOG", "$SPX", "XOM", "msft"),
    start=datetime.datetime(2005, 1, 1),
    end=datetime.datetime(2011, 12, 31),  # data stops at 2013/1/1
    normalize=True,
    allocation=None, 
    price_type='actual_close',
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

    ls_keys = [price_type]
    ldf_data = da.get_data(timestamps, symbols, ls_keys)
    d_data = dict(zip(ls_keys, ldf_data))

    na_price = d_data[price_type].values
    if normalize:
        na_price /= na_price[0, :]
    na_price *= allocation
    return np.sum(na_price, axis=1)

COLUMN_SEP = re.compile(r'[,/;]')

def make_dataframe(prices, num_prices=1, columns=('portfolio',)):
    """Convert a file, list of strings, or list of tuples into a Pandas DataFrame

    Arguments:
      num_prices (int): if not null, the number of columns (from right) that contain numeric values
    """
    if isinstance(prices, (pd.DataFrame, pd.Series)):
        return prices
    if isinstance(prices, basestring) and os.path.isfile(prices):
        prices = open(prices, 'rU')
    if isinstance(prices, file):
        values = []
        # FIXME: what if it's not a CSV but a TSV or PSV
        csvreader = csv.reader(prices, dialect='excel', quoting=csv.QUOTE_MINIMAL)
        for row in csvreader:
            # print row
            values += [row]
        prices.close()
        prices = values
    if isinstance(prices[0], basestring):
        prices = [COLUMN_SEP.split(row) for row in prices]
    # print prices
    index = []
    if isinstance(prices[0][0], (datetime.date, datetime.datetime, datetime.time)):
        index = [prices[0] for row in prices]
        for i, row in prices:
            prices[i] = row[1:]
    # try to convert all strings to something numerical:
    elif any(any(isinstance(value, basestring) for value in row) for row in prices):
        #print '-'*80
        for i, row in enumerate(prices):
            #print i, row
            for j, value in enumerate(row):
                s = unicode(value).strip().strip('"').strip("'")
                #print i, j, s
                try:
                    prices[i][j] = int(s)
                    # print prices[i][j]
                except:
                    try:
                        prices[i][j] = float(s)
                    except:
                        # print 'FAIL'
                        try:
                            # this is a probably a bit too forceful
                            prices[i][j] = parse_date(s)
                        except:
                            pass
    # print prices
    width = max(len(row) for row in prices)
    datetime_width = width - num_prices
    if not index and isinstance(prices[0], (tuple, list)) and num_prices:
        # print '~'*80
        new_prices = []
        try:
            for i, row in enumerate(prices):
                # print i, row
                index += [datetime.datetime(*[int(i) for i in row[:datetime_width]])
                          + datetime.timedelta(hours=16)]
                new_prices += [row[datetime_width:]]
                # print prices[-1]
        except:
            for i, row in enumerate(prices):
                index += [row[0]]
                new_prices += [row[1:]]
        prices = new_prices or prices
    # print index
    # TODO: label the columns somehow (if first row is a bunch of strings/header)
    if len(index) == len(prices):
        df = pd.DataFrame(prices, index=index, columns=columns)
    else:
        df = pd.DataFrame(prices)
    return df


def metrics(prices, fudge=False, sharpe_days=252., baseline='$SPX'):
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
        values = {}
        csvreader = csv.reader(prices, dialect='excel', quoting=csv.QUOTE_MINIMAL)
        for row in csvreader:
            # print row
            values[tuple(int(s) for s in row[:3])] = row[-1]
        prices.close()
        prices = [v for (k,v) in sorted(values.items())]
        print prices
    if isinstance(prices[0], (tuple, list)):
        prices = [row[-1] for row in prices]
    if sharpe_days == None:
        sharpe_days = len(prices)
    prices = np.array([float(p) for p in prices])
    if not isinstance(fudge, bool) and fudge:
        fudge = float(fudge)
    elif fudge == True or (isinstance(fudge, float) and fudge == 0.0):
        fudge = (len(prices) - 1.) / len(prices)
    else:
        fudge = 1.0
    daily_returns = np.diff(prices) / prices[0:-1]
    # print daily_returns
    end_price = float(prices[-1])
    start_price = (prices[0])
    mean = fudge * np.average(daily_returns)
    variance = fudge * np.sum((daily_returns - mean) * (daily_returns - mean)) / float(len(daily_returns))
    results = {
        'standared deviation of daily returns': math.sqrt(variance), 
        'variance of daily returns': variance, 
        'average daily return': mean, 
        'Sharpe ratio': mean * np.sqrt(sharpe_days) / np.sqrt(variance), 
        'total return': end_price / start_price,
        'final value': end_price,
        'starting value': start_price, 
        }
    results['return rate'] = results['total return'] - 1.0
    return results


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



def analyze(args):
    print 'Report for {0}...'.format(args.infile)
    report = metrics(args.infile, fudge=args.fudge, sharpe_days=args.sharpe_days)
    print report

    return report

def event(args):
    return buy_on_drop(symbol_set=args.symbols, 
            dataobj=dataobj, 
            start=args.start, 
            end=args.end,
            market_sym=args.baseline,
            threshold=args.threshold,
            sell_delay=args.delay)


##############################################
## Event Studies

def event_happened(**kwargs):
    """Function that takes as input various prices (today, yesterday, etc) and returns True if an "event" has been triggered

    Examples:
        Event is found if the symbol is down more then 3% while the market is up more then 2%:
        return bool(kwargs['return_today'] <= -0.03 and kwargs['market_return_today'] >= 0.02)
    """
    return bool(kwargs['price_today'] < 8.0 and kwargs['price_yest'] >= 8.0)


def drop_below(threshold=5, **kwargs):
    """Trigger function that returns True if the price falls below the threshold

    price_today < threshold and price_yest >= threshold
    """
    if (
    #    kwargs['price_today'] and kwargs['price_yest'] and
    #    not np.isnan(kwargs['price_today'] and not kwargs['price_yest'] and
        kwargs['price_today'] < threshold and kwargs['price_yest'] >= threshold
        ):
        return True
    else:
        return False

def generate_orders(events, sell_delay=5, sep=','):
    """Generate CSV orders based on events indicated in a DataFrame

    Arguments:
      events (pandas.DataFrame): Table of NaNs or 1's, one column for each symbol.
        1 indicates a BUY event. -1 a SELL event. nan or 0 is a nonevent.
      sell_delay (float): Number of days to wait before selling back the shares bought
      sep (str or None): if sep is None, orders will be returns as tuples of `int`s, `float`s, and `str`s
        otherwise the separator will be used to join the order parameters into the yielded str

    Returns:
       generator of str: yielded CSV rows in the format (yr, mo, day, symbol, Buy/Sell, shares)
    """
    sell_delay = float(unicode(sell_delay)) or 1
    for i, (t, row) in enumerate(events.iterrows()):
        for sym, event in row.to_dict().iteritems():
            # print sym, event, type(event)
            # return events
            if event and not np.isnan(event):
                # add a sell event `sell_delay` in the future within the existing `events` DataFrame
                # modify the series, but only in the future and be careful not to step on existing events
                if event > 0:
                    sell_event_i = min(i + sell_delay, len(events) - 1)
                    sell_event_t = events.index[sell_event_i]
                    sell_event = events[sym][sell_event_i]
                    if np.isnan(sell_event):
                        events[sym][sell_event_t] = -1
                    else:
                        events[sym][sell_event_t] += -1
                order = (t.year, t.month, t.day, sym, 'Buy' if event > 0 else 'Sell', abs(event) * 100)
                if isinstance(sep, basestring):
                    yield sep.join(order)
                yield order

@memoize
def find_events(symbols, d_data, market_sym='$SPX', trigger=drop_below, trigger_kwargs={}):
    '''Return dataframe of 1's (event happened) and NaNs (no event), 1 column for each symbol'''

    df_close = d_data['actual_close']
    ts_market = df_close[market_sym]

    print "Finding `{0}` events with kwargs={1} for {2} ticker symbols".format(trigger.func_name, trigger_kwargs, len(symbols))
    print 'Trigger docstring says:\n\n{0}\n\n'.format(trigger.func_doc)

    # Creating an empty dataframe
    df_events = copy.deepcopy(df_close)
    df_events = df_events * np.NAN

    # Time stamps for the event range
    ldt_timestamps = df_close.index

    for s_sym in symbols:
        if s_sym == market_sym:
            continue
        for i in range(1, len(ldt_timestamps)):
            # Calculating the returns for this timestamp
            kwargs = dict(trigger_kwargs)
            kwargs['price_today'] = df_close[s_sym].ix[ldt_timestamps[i]]
            kwargs['price_yest'] = df_close[s_sym].ix[ldt_timestamps[i - 1]]
            kwargs['return_today'] = (kwargs['price_today'] / (kwargs['price_yest'] or 1.)) - 1
            kwargs['market_price_today'] = ts_market.ix[ldt_timestamps[i]]
            kwargs['market_price_yest'] = ts_market.ix[ldt_timestamps[i - 1]]
            kwargs['market_return_today'] = (kwargs['market_price_today'] / (kwargs['market_price_yest'] or 1.)) - 1

            if trigger(**kwargs):
                df_events[s_sym].ix[ldt_timestamps[i]] = 1
    print 'Found {0} events where priced dropped below {1}.'.format(df_events.sum(axis=1).sum(axis=0), trigger_kwargs['threshold'])
    return df_events

def get_clean_data(symbols=None, 
                   dataobj=dataobj, 
                   start=None, 
                   end=datetime.datetime(2009, 12, 31),
                   market_sym='$SPX',
                   reset_cache=True):
    start = start or datetime.datetime(2008, 1, 1)
    end = end or datetime.datetime(2009, 12, 31)
    if not symbols:
        symbols = dataobj.get_symbols_from_list("sp5002012")
        symbols.append(market_sym)


    print "Calculating timestamps for {0} SP500 symbols".format(len(symbols))
    ldt_timestamps = du.getNYSEdays(start, end, datetime.timedelta(hours=16))

    ls_keys = ['open', 'high', 'low', 'close', 'volume', 'actual_close']
    print "Retrieving data for {0} SP500 symbols between {1} and {2}.".format(len(symbols), start, end)
    ldf_data = dataobj.get_data(ldt_timestamps, symbols, ls_keys, )
    d_data = dict(zip(ls_keys, ldf_data))

    for s_key in ls_keys:
        print 'cleaning nans from the column {0}'.format(repr(s_key))
        d_data[s_key] = d_data[s_key].fillna(method='ffill')
        d_data[s_key] = d_data[s_key].fillna(method='bfill')
        d_data[s_key] = d_data[s_key].fillna(1.0)
    return d_data


def buy_on_drop(symbol_set="sp5002012", 
            dataobj=dataobj, 
            start=datetime.datetime(2008, 1, 3), 
            end=datetime.datetime(2009, 12, 28),
            market_sym='$SPX',
            threshold=6,
            sell_delay=5,
            ):
    '''Compute and display an "event profile" for multiple sets of symbols'''
    if symbol_set:
        if isinstance(symbol_set, basestring):
            if symbol_set.lower().startswith('sp'):
                symbol_set = dataobj.get_symbols_from_list(symbol_set.lower())
            else:
                symbol_set = [sym.stip().upper() for sym in symbol_set.split(",")]
    else:
        symbol_set = dataobj.get_symbols_from_list("sp5002012")
    if market_sym:
        symbol_set.append(market_sym)

    print "Starting Event Study, retrieving data for the {0} symbol list...".format(symbol_set)
    market_data = get_clean_data(symbol_set, dataobj=dataobj, start=start, end=end)
    print "Finding events for {0} symbols between {1} and {2}...".format(len(symbol_set), start, end)
    trigger_kwargs={'threshold': threshold}
    events = find_events(symbol_set, market_data,  market_sym=market_sym, trigger=drop_below, trigger_kwargs=trigger_kwargs)

    csvwriter = csv.writer(getattr(args, 'outfile', open('buy_on_drop_outfile.csv', 'w')), dialect='excel', quoting=csv.QUOTE_MINIMAL)
    for order in generate_orders(events, sell_delay=sell_delay, sep=None):
        csvwriter.writerow(order)

    print "Creating Study report for {0} events...".format(len(events))
    ep.eventprofiler(events, market_data, 
                         i_lookback=20, i_lookforward=20,
                         s_filename='Event report--buy on drop below {0} for {1} symbols.pdf'.format(threshold, len(symbol_set)),
                         b_market_neutral=True,
                         b_errorbars=True,
                         s_market_sym=market_sym,
                         )
    return events

## Event Studies
##############################################


def build_args_parser(parser=None):
    # create the top-level parser for this "sim" module
    parser = argparse.ArgumentParser(prog='sim', description='Simulate trading and predictive analytics algorithms.')
    parser.add_argument('--source',
                        default='Yahoo',
                        choices=('Yahoo', 'Google', 'Bloomberg'),
                        help='Name of financial data source to use in da.DataAccess("Name")')

    subparsers = parser.add_subparsers(help='`sim command` help')

    # create the parser for the "trade" command
    parser_trade = subparsers.add_parser('trade', help='Simulate a sequence of trades')
    parser_trade.add_argument('funds', type=float,
                              nargs='?',
                              default=50000.0,
                              help='Initial funds (cash, USD) in portfolio.')
    parser_trade.add_argument('infile', nargs='?', type=argparse.FileType('rU'),
                              help='Path to input CSV file containing a list of trades: y,m,d,sym,BUY/SELL,shares',
                              default=sys.stdin)
    parser_trade.add_argument('outfile', nargs='?', type=argparse.FileType('w'),
                              help='Path to output CSV file where a time series of dates and portfolio values will be written',
                              default=sys.stdout)
    parser_trade.add_argument('--detailed', action='store_true',
                              help='Whether to output a json string containing the portfolio allocation as the 2nd-to-last CSV column')
    parser_trade.add_argument('--trades', action='store_true',
                              help='Whether to output the buy/sell events along with the total value of the portfolio in the output.')
    parser_trade.add_argument('--simple', action='store_true',
                              help='Whether to supress output of a json string containing the portfolio allocation as the 2nd-to-last CSV column')
    parser_trade.set_defaults(func=trade)


    # create the parser for the "analyze" command
    parser_analyze = subparsers.add_parser('analyze', help='Analyze a time series (sequence) of prices (typically portfolio values)')
    parser_analyze.add_argument('infile', nargs='?', type=argparse.FileType('rU'),
                              help='Path to input CSV file containing sequence of prices (portfolio values) in the last column. Typically each line should be a (yr, mo, dy, price) CSV string for each trading day in the sequence',
                              default=sys.stdin)
    parser_analyze.add_argument('--fudgefactor', type=float, default=False,
                              help="Value to multiply the total return by to make it match Tucker Balche's incorrect math")
    parser_analyze.add_argument('--fudge', action='store_true',
                              help="Whether to fudge by N-1")
    parser_analyze.add_argument('--sharpe_days', type=float, default=252.0,
                              help="Value to multiple the daily average return by to get the yearly return for Sharpe ratio calculation.")
    parser_analyze.set_defaults(func=analyze)


    # create the parser for the "event" command (event studies)
    parser_event = subparsers.add_parser('event', help='Generate a sequence of trades based on an event study (trigger events)')
    parser_event.add_argument('outfile', nargs='?', type=argparse.FileType('w'),
                              help="Path to output CSV file to contain the trades (yr, mo, day, symbol, 'Buy'/'Sell', shares)",
                              default=sys.stdout)
    parser_event.add_argument('--price', type=str, default="actual_close",
                              help="Which price to trigger on (close, actual_close, volume)")
    parser_event.add_argument('--threshold', type=float, default=5.0,
                              help="Buy equities whenever they fall below this actual_close price")
    parser_event.add_argument('--start', type=parse_date, default='2008-01-01',
                              help="Start of time period to perform event study.")
    parser_event.add_argument('--end', type=parse_date, default='2009-12-31',
                              help="End of time period to perform event study.")
    parser_event.add_argument('--delay', type=float, default=5,
                              help="Number of days to hold the stock before selling it.")
    parser_event.add_argument('--symbols', type=str, default="sp5002012",
                              help="Which stocks to search for events for (sp5002012, sp5002008, all, ...).")
    parser_event.add_argument('--baseline', type=str, default="$SPX",
                              help="Which stocks to search for events for (sp5002012, sp5002008, all, ...).")
    parser_event.set_defaults(func=event)

    # print sys.argv
    # #args = parser.parse_args()
    # argsv1 = ' '.join(sys.argv[1:]).split()
    # print argsv1
    # argsv2 = ' '.join('analyze 1 -x 2'.split()).split()
    # print argsv2
    # args2 = parser.parse_args(argsv2)
    # args.func(args2)
    # print dir(args2)
    # args1 = parser.parse_args(argsv1)
    # args.func(args1)
    # print dir(args1)

    return parser


if __name__ == '__main__':
    # build the parser and then use it to parse the arguments in sys.args
    args = build_args_parser().parse_args()
    # run `sim()` or `analyze()` or whatever function is indicated by the `subparser.set_defaults()` for `args.main` 
    args.func(args)

