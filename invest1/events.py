#!/usr/bin/env python
'''Finds events in time-series data and plots statistics about following and preceding events'''
from __future__ import unicode_literals
from __future__ import division

import sys
import os
import copy
import datetime as dt
import argparse
import csv
# import math

import numpy as np
#import pandas as pd
import QSTK.qstkutil.qsdateutil as du
import QSTK.qstkutil.DataAccess as da
# import QSTK.qstkutil.tsutil as tsu
import QSTK.qstkstudy.EventProfiler as ep

from pug.decorators import memoize

PROG = os.path.splitext(__file__)[0]

tucker = da.DataAccess('Yahoo')

def event_happened(**kwargs):
    """Function that takes as input various prices (today, yesterday, etc) and returns True if an "event" has been triggered

    Examples:
        Event is found if the symbol is down more then 3% while the market is up more then 2%:
        return bool(kwargs['return_today'] <= -0.03 and kwargs['market_return_today'] >= 0.02)
    """
    return bool(kwargs['price_today'] < 8.0 and kwargs['price_yest'] >= 8.0)


def drop_below(threshold=5, **kwargs):
    """Trigger function that returns True if the price falls below the threshold

    price_today < threshold and price_yesterday >= threshold
    """
    if (
    #    kwargs['price_today'] and kwargs['price_yest'] and
    #    not np.isnan(kwargs['price_today'] and not kwargs['price_yest'] and
        kwargs['price_today'] < threshold and kwargs['price_yest'] >= threshold
        ):
        return True
    else:
        return False


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
                   dataobj=None, 
                   start=None, 
                   end=dt.datetime(2009, 12, 31),
                   market_sym='$SPX',
                   reset_cache=True):
    dataobj = dataobj or tucker
    start = start or dt.datetime(2008, 1, 1)
    end = end or dt.datetime(2009, 12, 31)
    if not symbols:
        symbols = dataobj.get_symbols_from_list("sp5002008")
        symbols.append(market_sym)


    print "Calculating timestamps for {0} SP500 symbols".format(len(symbols))
    ldt_timestamps = du.getNYSEdays(start, end, dt.timedelta(hours=16))

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


def compare(symbol_sets=None, 
            dataobj=None, 
            start=None, 
            end=None,
            market_sym='$SPX',
            threshold=5,
            ):
    '''Compute and display an "event profile" for multiple sets of symbols'''
    if not symbol_sets:
        symbol_sets = {2008: dataobj.get_symbols_from_list("sp5002008"), 2012: dataobj.get_symbols_from_list("sp5002012")}
        symbol_sets[2008].append(market_sym), symbol_sets[2012].append(market_sym)

    print "Starting Event Study..."
    event_profiles = []
    for yr, symbols in symbol_sets.iteritems():
        print "Cleaning NaNs from data for {0} symbols in SP500-{1}...".format(len(symbols), yr)
        d_data = get_clean_data(symbols, dataobj=dataobj, start=start, end=end)
        print "Finding events for {0} symbols in SP500-{1}...".format(len(symbols), yr)
        df_events = find_events(symbols, d_data, threshold=threshold)
        print "Creating Study report for {0} events...".format(len(df_events))
        event_profiles += [
            ep.eventprofiler(df_events, d_data, 
                             i_lookback=20, i_lookforward=20,
                             s_filename='event_study_report-10dollar-{0}.pdf'.format(yr),
                             b_market_neutral=True,
                             b_errorbars=True,
                             s_market_sym=market_sym,
                             )]
    return event_profiles


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
    sell_delay = float(unicode(sell_delay))
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


def buy_on_drop(args, symbol_set=None, 
            dataobj=tucker, 
            start=dt.datetime(2008, 1, 3), 
            end=dt.datetime(2009, 12, 28),
            market_sym='$SPX',
            threshold=5,
            yr=2012,
            sell_delay=5,
            ):
    '''Compute and display an "event profile" for multiple sets of symbols'''
    if not symbol_set:
        symbol_set = dataobj.get_symbols_from_list("sp500{0}".format(yr))
        symbol_set.append(market_sym)

    print "Starting Event Study, retrieving data..."
    market_data = get_clean_data(symbol_set, dataobj=dataobj, start=start, end=end)
    print "Finding events for {0} symbols in SP500-{1}...".format(len(symbol_set), yr)
    trigger_kwargs={'threshold': threshold}
    events = find_events(symbol_set, market_data,  market_sym=market_sym, trigger=drop_below, trigger_kwargs=trigger_kwargs)

    csvwriter = csv.writer(args.outfile, dialect='excel', quoting=csv.QUOTE_MINIMAL)
    for order in generate_orders(events, sell_delay=sell_delay, sep=None):
        csvwriter.writerow(order)

    print "Creating Study report for {0} events...".format(len(events))
    ep.eventprofiler(events, market_data, 
                         i_lookback=20, i_lookforward=20,
                         s_filename='Event report--buy on drop below {1} for SP500-{0}.pdf'.format(yr, threshold),
                         b_market_neutral=True,
                         b_errorbars=True,
                         s_market_sym=market_sym,
                         )
    return events


def parse_args():
    "Create and run the top-level parser for this `{0}` module".format(PROG)

    parser = argparse.ArgumentParser(prog='events', description='Process market data to identify and act on trigger events.')
    parser.add_argument('--source',
                        default='Yahoo',
                        choices=('Yahoo', 'Google', 'Bloomberg'),
                        help='Name of financial data source to use in da.DataAccess("Name")')

    subparsers = parser.add_subparsers(help='`{0} trade` help'.format(PROG))

    # create the parser for the "a" command
    parser_trade = subparsers.add_parser('trade', help='Generate a sequence of trades based on event triggers')
    parser_trade.add_argument('outfile', nargs='?', type=argparse.FileType('w'),
                              help="Path to output CSV file to contain the trades (yr, mo, day, symbol, 'Buy'/'Sell', shares)",
                              default=sys.stdout)
    return parser.parse_args()

if __name__ == '__main__':
    """year threshold # events
       2012 6.0 220-230
       2008 8.0 510-530 or 540-550
    """
    args = parse_args()
    print args
    buy_on_drop(args)
    #print buy_on_drop()
