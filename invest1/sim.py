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

def main(args):
    print args


if __name__ == '__main__':

    # create the top-level parser for this "sim" module
    parser = argparse.ArgumentParser(prog='sim', description='Simulate trading and predictive analytics algorithms.')
    parser.add_argument('source',
                        default='Yahoo',
                        choices=('Yahoo', 'Google', 'Bloomberg'),
                        help='Name of financial data source to use in da.DataAccess("Name")')

    subparsers = parser.add_subparsers(help='`sim trade` help')

    # create the parser for the "a" command
    parser_trade = subparsers.add_parser('trade', help='Simulate a sequence of trades')
    parser_trade.add_argument('funds', type=float, 
                              default=1000000.,
                              help='Initial funds (cash, USD) in portfolio.')
    parser_trade.add_argument('infile', type=argparse.FileType('r'),
                              help='Path to input CSV file containing a list of trades: y,m,d,sym,BUY/SELL,shares',
                              default=sys.stdin)
    parser_trade.add_argument('outfile', nargs='?', type=argparse.FileType('w'),
                              help='Path to output CSV file containing a list of values of the portfolio over time',
                              default=sys.stdout)

    args = parser.parse_args()
    print args
