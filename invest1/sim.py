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
import argparse
import sys
import csv
import datetime as dt
import re

import numpy as np

from pug import debug
import QSTK.qstkutil.DataAccess as da
#/import QSTK.qstkutil.qsdateutil as du

import portfolio as report


#t = qstk.dateutil.getNYSEdays(datetime.datetime(2010,1,1), datetime.datetime(2010,2,1), dt.timedelta(hours=16))
dataobj = da.DataAccess('Yahoo')

DATE_SEP = re.compile(r'[^0-9]')

def get_price(symbol='SPY', date=(2010,1,1), price='actual_close'):
    # if hasattr(symbol, '__iter__'):
    #     return [get_price(sym, date=date, price=price) for sym in symbol]
    if isinstance(date, basestring):
        date = DATE_SEP.split(date)
    if isinstance(date, (tuple, list)):
        date = dt.datetime(*[int(i) for i in date])
    if isinstance(date, dt.date) or not (9 <= date.hour <= 16):
        date = dt.datetime(date.year, date.month, date.day, 16)
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

    #trading_days = du.getNYSEdays(dt.datetime(2010,01,01), dt.datetime(2012,01,01), dt.timedelta(hours=16))
    for row in csvreader:
        print '-'*30 + ' CSV Row ' + '-'*30
        print ', '.join(row)
        trade_date = dt.datetime(*[int(i) for i in (row[:3] + [16])])

        if history:
            last_date = dt.datetime(*(history[-1][:3] + [16])) + dt.timedelta(1)
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
                last_date += dt.timedelta(1)
    
        trade_symbol = row[3]
        trade_shares = float(row[5])
        trade_sign = 1 - 2 * int(row[4].strip().upper()[0]=='S')
        # print date, symbol, sign * shares
        portfolio[trade_symbol] = portfolio.get(trade_symbol, 0) + trade_sign * trade_shares
        trade_price = get_price(symbol=trade_symbol, date=trade_date, price='actual_close')
        while trade_price == None or np.isnan(trade_price) or float(trade_price) == float('nan') or float(trade_price) == None:
            trade_date += dt.timedelta(1)
            trade_price = get_price(symbol=trade_symbol, date=trade_date, price='actual_close')
        print trade_date, trade_symbol, trade_sign, trade_shares, trade_price
        if trade_price and trade_shares and trade_sign in (-1, 1):
            portfolio['$CASH'] -= trade_sign * trade_shares * trade_price
        else:
            print 'ERROR: bad price, sign, shares: ', trade_price, trade_sign, trade_shares
        history += [[trade_date.year, trade_date.month, trade_date.day, portfolio_value(portfolio, trade_date)]]
        csvwriter.writerow(history[-1])
    print report.metrics(history)


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

