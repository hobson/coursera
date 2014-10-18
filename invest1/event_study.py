'''
(c) 2011, 2012 Georgia Tech Research Corporation
This source code is released under the New BSD license.  Please see
http://wiki.quantsoftware.org/index.php?title=QSTK_License
for license details.

Created on January, 23, 2013

@author: Sourabh Bajaj
@contact: sourabhbajaj@gatech.edu
@summary: Event Profiler Tutorial
'''


#import math
import copy
import datetime as dt

import numpy as np
#import pandas as pd
import QSTK.qstkutil.qsdateutil as du
import QSTK.qstkutil.DataAccess as da
# import QSTK.qstkutil.tsutil as tsu
import QSTK.qstkstudy.EventProfiler as ep

"""
Accepts a list of symbols along with start and end date
Returns the Event Matrix which is a pandas Datamatrix
Event matrix has the following structure :
    |IBM |GOOG|XOM |MSFT| GS | JP |
(d1)|nan |nan | 1  |nan |nan | 1  |
(d2)|nan | 1  |nan |nan |nan |nan |
(d3)| 1  |nan | 1  |nan | 1  |nan |
(d4)|nan |  1 |nan | 1  |nan |nan |
...................................
...................................
Also, d1 = start date
nan = no information about any event.
1 = status bit(positively confirms the event occurence)
"""


dataobj = da.DataAccess('Yahoo')

def event_happened(**kwargs):
    """Function that takes as input various prices (today, yesterday, etc) and returns True if an "event" has been triggered

    Examples:
        Event is found if the symbol is down more then 3% while the market is up more then 2%:
        return bool(kwargs['return_today'] <= -0.03 and kwargs['market_return_today'] >= 0.02)
    """
    return bool(kwargs['price_today'] < 8.0 and kwargs['price_yest'] >= 8.0)

def drop_below(threshold=5, **kwargs):
    """Function that takes as input various prices (today, yesterday, etc) and returns True the price falls below the threshold
    """
    return bool(kwargs['price_today'] < threshold and kwargs['price_yest'] >= threshold)


def find_events(ls_symbols, d_data, market_sym='$SPX', threshold=5):
    ''' Finding Events to put in a dataframe '''

    df_close = d_data['actual_close']
    ts_market = df_close[market_sym]

    print "Finding events where price dropped below {1} for {0} symbols".format(len(ls_symbols), threshold)

    # Creating an empty dataframe
    df_events = copy.deepcopy(df_close)
    df_events = df_events * np.NAN

    # Time stamps for the event range
    ldt_timestamps = df_close.index

    for s_sym in ls_symbols:
        if s_sym == market_sym:
            continue
        for i in range(1, len(ldt_timestamps)):
            # Calculating the returns for this timestamp
            kwargs = {}
            kwargs['price_today'] = df_close[s_sym].ix[ldt_timestamps[i]]
            kwargs['price_yest'] = df_close[s_sym].ix[ldt_timestamps[i - 1]]
            kwargs['return_today'] = (kwargs['price_today'] / (kwargs['price_yest'] or 1.)) - 1
            kwargs['market_price_today'] = ts_market.ix[ldt_timestamps[i]]
            kwargs['market_price_yest'] = ts_market.ix[ldt_timestamps[i - 1]]
            kwargs['market_return_today'] = (kwargs['market_price_today'] / (kwargs['market_price_yest'] or 1.)) - 1

            if drop_below(threshold=threshold, **kwargs):
                df_events[s_sym].ix[ldt_timestamps[i]] = 1
    print 'Found {0} events where priced dropped below {1}.'.format(df_events.sum(axis=1).sum(axis=0), threshold)
    return df_events


def get_clean_data(symbols=None, 
                   dataobj=da.DataAccess('Yahoo'), 
                   start=dt.datetime(2008, 1, 1), 
                   end=dt.datetime(2009, 12, 31),
                   market_sym='$SPX',
                   reset_cache=True):
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
            dataobj=da.DataAccess('Yahoo'), 
            start=dt.datetime(2008, 1, 1), 
            end=dt.datetime(2009, 12, 31),
            market_sym='$SPX',
            threshold=5,
            ):
    '''Compute and display an "event profile" for multiple sets of symbols'''
    if not symbol_sets:
        symbol_sets = {2008: dataobj.get_symbols_from_list("sp5002008"), 2012: dataobj.get_symbols_from_list("sp5002012")}
        symbol_sets[2008].append(market_sym), symbol_sets[2012].append(market_sym)

    print "Starting Event Study..."
    event_profiles = []
    for yr, ls_symbols in symbol_sets.iteritems():
        print "Cleaning NaNs from data for {0} symbols in SP500-{1}...".format(len(ls_symbols), yr)
        d_data = get_clean_data(ls_symbols, dataobj=dataobj, start=start, end=end)
        print "Finding events for {0} symbols in SP500-{1}...".format(len(ls_symbols), yr)
        df_events = find_events(ls_symbols, d_data, threshold=threshold)
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


def buy_on_drop(
            symbol_set=None, 
            dataobj=da.DataAccess('Yahoo'), 
            start=dt.datetime(2008, 1, 1), 
            end=dt.datetime(2009, 12, 31),
            market_sym='$SPX',
            threshold=5,
            yr=2012,
            ):
    '''Compute and display an "event profile" for multiple sets of symbols'''
    if not symbol_set:
        symbol_set = dataobj.get_symbols_from_list("sp500{0}".format(yr))
        symbol_set.append(market_sym)

    print "Starting Event Study..."
    print "Cleaning NaNs from data for {0} symbols in SP500-{1}...".format(len(symbol_set), yr)
    market_data = get_clean_data(symbol_set, dataobj=dataobj, start=start, end=end)
    print "Finding events for {0} symbols in SP500-{1}...".format(len(symbol_set), yr)
    events = find_events(symbol_set, market_data, threshold=threshold, market_sym=market_sym)
    for row in events:
        print row
    print "Creating Study report for {0} events...".format(len(events))
    # event_profile = ep.eventprofiler(df_events, d_data, 
    #                      i_lookback=20, i_lookforward=20,
    #                      s_filename='event_study_report-10dollar-{0}.pdf'.format(yr),
    #                      b_market_neutral=True,
    #                      b_errorbars=True,
    #                      s_market_sym=market_sym,
    #                      )
    #return event_profile

if __name__ == '__main__':
    """year threshold # events
       2012 6.0 220-230
       2008 8.0 510-530 or 540-550
    """
    print buy_on_drop()
    # print compare()
