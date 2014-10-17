#!/usr/bin/env python
# Example command line usage:
# python marketsim.py 1000000 orders.csv values.csv
#
# Example orders.csv:
# 2008, 12, 3, AAPL, BUY, 130
# 2008, 12, 8, AAPL, SELL, 130
# 2008, 12, 5, IBM, BUY, 50
#
# Example output file values.csv
# 2008, 12, 3, 1000000
# 2008, 12, 4, 1000010
# 2008, 12, 5, 1000250

import argparse
import sys
import csv
import datetime as dt

import numpy as np

import QSTK.qstkutil.DataAccess as da
import QSTK.qstkutil.qsdateutil as du
#t = qstk.dateutil.getNYSEdays(datetime.datetime(2010,1,1), datetime.datetime(2010,2,1), dt.timedelta(hours=16))
dataobj = da.DataAccess('Yahoo')


def get_price(symbol='SPY', date=(2010,1,1), price='close'):
    # if hasattr(symbol, '__iter__'):
    #     return [get_price(sym, date=date, price=price) for sym in symbol]
    if isinstance(date, basestring):
        date = dt.datetime(*date.split(','))
    elif isinstance(date, (tuple, list)):
        date = dt.datetime(*date)
    elif isinstance(date, dt.date):
        date = dt.datetime(date.year, date.month, date.day, 16)
    symbol = str(symbol).upper().strip()
    if symbol == '$CASH':
        return 1.
    try:
        return dataobj.get_data([date], [symbol], 'close')[symbol][0]
    except:
        print 'BAD DATE ({0}) or SYMBOL ({1})'.format(date, symbol) 
        return None


def main(args):
    print args
    print vars(args)['funds']
    print args.funds
    portfolio = { '$CASH': args.funds }
    print portfolio
    csvreader = csv.reader(args.infile, dialect='excel', quoting=csv.QUOTE_MINIMAL)
    history = []

    trading_days = du.getNYSEdays(dt.datetime(2010,01,01), dt.datetime(2012,01,01), dt.timedelta(hours=16))
    for row in csvreader:
        csvwriter = csv.writer(args.outfile, dialect='excel', quoting=csv.QUOTE_MINIMAL)
        print '-'*80
        print ', '.join(row)
        symbol = row[3]
        shares = float(row[5])
        date = dt.datetime(*[int(i) for i in row[:3] + ['16']])
        sign = 1 - 2 * int(row[4].strip().upper()[0]=='S')
        # print date, symbol, sign * shares
        portfolio[symbol] = portfolio.get(symbol, 0) + sign * shares
        trade_price = get_price(symbol=symbol, date=date, price='close')
        while trade_price == None or np.isnan(trade_price) or float(trade_price) == float('nan') or float(trade_price) == None:
            date += dt.timedelta(1)
            trade_price = get_price(symbol=symbol, date=date, price='close')
        print date, symbol, sign, shares, trade_price

        if history:
            last_date = dt.date(*history[-1][:3])
            # print (date.date() - last_date).days
            while (date.date() - last_date).days > 1:
                print 'filling in the blanks for {0}'.format(last_date)
                last_date += dt.timedelta(1)
                value = 0.
                for (k, v) in portfolio.iteritems():
                    price = get_price(symbol=k, date=last_date, price='close')
                    print k, v, price
                    # print last_date, k, price
                    if price != None and not np.isnan(price):
                        value = value + (float(v) * float(price))
                        print 'new price, value = {0}, {1}'.format(price, value)
                    else:
                        # print 'NAN'*20
                        value = None
                        break
                if value != None and not np.isnan(value):
                    print '='*20 + str(value)
                    history += [[last_date.year, last_date.month, last_date.day, value]]
                    csvwriter.writerow(history[-1])
    
        if trade_price and shares and sign in (-1, 1):
            portfolio['$CASH'] -= sign * shares * trade_price
        else:
            print 'ERROR: bad price, sign, shares: ', price, sign, shares
        total = 0.
        for (k, v) in portfolio.iteritems():
            price = get_price(symbol=k, date=date, price='close')
            # print date, k, price
            total += v * price
        print portfolio

        history += [[date.year, date.month, date.day, total]]
        csvwriter.writerow(history[-1])


if __name__ == '__main__':

    # create the top-level parser for this "sim" module
    parser = argparse.ArgumentParser(prog='sim', description='Simulate trading and predictive analytics algorithms.')
    parser.add_argument('--source',
                        default='Yahoo',
                        choices=('Yahoo', 'Google', 'Bloomberg'),
                        help='Name of financial data source to use in da.DataAccess("Name")')

    subparsers = parser.add_subparsers(help='`sim trade` help')

    # create the parser for the "a" command
    parser_trade = subparsers.add_parser('trade', help='Simulate a sequence of trades')
    parser_trade.add_argument('funds', type=float,
                              nargs='?',
                              default=1000000.,
                              help='Initial funds (cash, USD) in portfolio.')
    parser_trade.add_argument('infile', nargs='?', type=argparse.FileType('rU'),
                              help='Path to input CSV file containing a list of trades: y,m,d,sym,BUY/SELL,shares',
                              default=sys.stdin)
    parser_trade.add_argument('outfile', nargs='?', type=argparse.FileType('w'),
                              help='Path to output CSV file containing a list of values of the portfolio over time',
                              default=sys.stdout)

    args = parser.parse_args()
    print args
    main(args)
