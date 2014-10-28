#!/usr/bin/env python
from __future__ import division, unicode_literals
"""Simulate a sequence of trades and portfolio performance over time

Examples:
  $ python marketsim.py 1000000 orders.csv values.csv

  # input file
  $ cat orders.csv:
  2008, 12, 3, AAPL, BUY, 130
  2008, 12, 8, AAPL, SELL, 130
  2008, 12, 5, IBM, BUY, 50

  # output file
  $ cat values.csv
  2008, 12, 3, 1000000
  2008, 12, 4, 1000010
  2008, 12, 5, 1000250
"""
from pug import debug
import argparse
import sys
import csv
import datetime
import re

import numpy as np
import pandas as pd

import QSTK.qstkutil.DataAccess as da
import QSTK.qstkutil.qsdateutil as du

import portfolio as report


#t = qstk.dateutil.getNYSEdays(datetime.datetime(2010,1,1), datetime.datetime(2010,2,1), datetime.timedelta(hours=16))
dataobj = da.DataAccess('Yahoo')

DATE_SEP = re.compile(r'[^0-9]')

def get_price(symbol='SPY', date=(2010,1,1), price='actual_close'):
    # if hasattr(symbol, '__iter__'):
    #     return [get_price(sym, date=date, price=price) for sym in symbol]
    if isinstance(date, basestring):
        date = DATE_SEP.split(date)
    if isinstance(date, (tuple, list)):
        date = datetime.datetime(*[int(i) for i in date])
    if isinstance(date, datetime.date) or not (9 <= date.hour <= 16):
        date = datetime.datetime(date.year, date.month, date.day, 16)
    symbol = str(symbol).upper().strip()
    if symbol == '$CASH':
        return 1.
    try:
        return dataobj.get_data([date], [symbol], [price])[0][symbol][0]
    except IndexError:
        raise
    except:
        print 'BAD DATE ({0}) or SYMBOL ({1})'.format(date, symbol) 
        return None


def portfolio_value(portfolio, date):
    """Total value of a portfolio (dict mapping symbols to numbers of shares)

    $CASH used as symbol for USD
    """
    value = 0.
    for (sym, sym_shares) in portfolio.iteritems():
        sym_price = None
        if sym_shares:
            sym_price = get_price(symbol=sym, date=date, price='actual_close')
        print sym, sym_shares, sym_price
        # print last_date, k, price
        if sym_price != None:
            if np.isnan(sym_price):
                print 'Invalid price, shares, value, total: ', sym_price, sym_shares, (float(sym_shares) * float(sym_price)) if sym_shares and sym_price else 'Invalid', value
                if sym_shares:
                    return float('nan')
            else:
                value += float(sym_shares) * float(sym_price)
                # print 'new price, value = {0}, {1}'.format(sym_price, value)
    return value


def sim(args):
    """Simulate a sequence of trades indicated in `infile` and write the portfolio value time series to `outfile`"""
    print args
    print vars(args)['funds']
    print args.funds
    portfolio = { '$CASH': args.funds }
    print portfolio
    csvreader = csv.reader(args.infile, dialect='excel', quoting=csv.QUOTE_MINIMAL)
    csvwriter = csv.writer(args.outfile, dialect='excel', quoting=csv.QUOTE_MINIMAL)
    history = []

    #trading_days = du.getNYSEdays(datetime.datetime(2010,01,01), datetime.datetime(2012,01,01), datetime.timedelta(hours=16))
    for row in csvreader:
        print '-'*30 + ' CSV Row ' + '-'*30
        print ', '.join(row)
        trade_date = datetime.datetime(*[int(i) for i in (row[:3] + [16])])

        if history:
            last_date = datetime.datetime(*(history[-1][:3] + [16])) + datetime.timedelta(1)
            # print (date.date() - last_date).days
            while (trade_date - last_date).days > 0:
                print 'Filling in the blanks for {0}'.format(last_date)
                value = portfolio_value(portfolio, last_date)
                print '   porfolio value on that date is: ' + str(value)
                assert(value != None)
                # NaN for porfolio value indicates a non-trading day
                if not np.isnan(value):
                    history += [[last_date.year, last_date.month, last_date.day, value]]
                    csvwriter.writerow(history[-1])
                last_date += datetime.timedelta(1)
    
        trade_symbol = row[3]
        trade_shares = float(row[5])
        trade_sign = 1 - 2 * int(row[4].strip().upper()[0]=='S')
        # print date, symbol, sign * shares
        portfolio[trade_symbol] = portfolio.get(trade_symbol, 0) + trade_sign * trade_shares
        trade_price = get_price(symbol=trade_symbol, date=trade_date, price='actual_close')
        while trade_price == None or np.isnan(trade_price) or float(trade_price) == float('nan') or float(trade_price) == None:
            trade_date += datetime.timedelta(1)
            trade_price = get_price(symbol=trade_symbol, date=trade_date, price='actual_close')
        print trade_date, trade_symbol, trade_sign, trade_shares, trade_price
        if trade_price and trade_shares and trade_sign in (-1, 1):
            portfolio['$CASH'] -= trade_sign * trade_shares * trade_price
        else:
            print 'ERROR: bad price, sign, shares: ', trade_price, trade_sign, trade_shares
        history += [[trade_date.year, trade_date.month, trade_date.day, portfolio_value(portfolio, trade_date)]]
        csvwriter.writerow(history[-1])
    print report.metrics(history)

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



def analyze(args):
    print 'Report for {0}'.format(args.infile)
    print report.metrics(args.infile)


def build_args_parser(parser=None):
    # create the top-level parser for this "sim" module
    parser = argparse.ArgumentParser(prog='sim', description='Simulate trading and predictive analytics algorithms.')
    parser.add_argument('--source',
                        default='Yahoo',
                        choices=('Yahoo', 'Google', 'Bloomberg'),
                        help='Name of financial data source to use in da.DataAccess("Name")')

    subparsers = parser.add_subparsers(help='`sim trade` help')
    print '?'*10 + 'subparser: '
    print subparsers
    print dir(parser)
    print parser.__dict__

    # create the parser for the "trade" command
    parser_trade = subparsers.add_parser('trade', help='Simulate a sequence of trades')
    parser_trade.add_argument('funds', type=float,
                              nargs='?',
                              default=1000000.,
                              help='Initial funds (cash, USD) in portfolio.')
    parser_trade.add_argument('infile', nargs='?', type=argparse.FileType('rU'),
                              help='Path to input CSV file containing a list of trades: y,m,d,sym,BUY/SELL,shares',
                              default=sys.stdin)
    parser_trade.add_argument('outfile', nargs='?', type=argparse.FileType('w'),
                              help='Path to output CSV file where a time series of dates and portfolio values will be written',
                              default=sys.stdout)
    parser_trade.set_defaults(func=sim)

    # create the parser for the "analyze" command
    parser_analyze = subparsers.add_parser('analyze', help='Analyze a time series (sequence) of prices (typically portfolio values)')
    parser_analyze.add_argument('infile', nargs='?', type=argparse.FileType('rU'),
                              help='Path to input CSV file containing sequence of prices (portfolio values) in the last column. Typically each line should be a (yr, mo, dy, price) CSV string for each trading day in the sequence',
                              default=sys.stdin)
    parser_analyze.set_defaults(func=analyze)

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

    print '!'*10 + 'subparser: '
    print subparsers
    print '!'*10 + 'dir(parser): '
    print dir(parser)
    print '!'*10 + 'parser.__dict: '
    print parser.__dict__
    return parser


if __name__ == '__main__':
    # build the parser and then use it to parse the arguments in sys.args
    args = build_args_parser().parse_args()
    # run `sim()` or `analyze()` or whatever function is indicated by the `subparser.set_defaults()` for `args.main` 
    args.func(args)

