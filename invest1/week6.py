#!/usr/bin/env python
from __future__ import division, unicode_literals
"""Attempt to duplicate the Homework 4 (week 6) example results

Threshold: $5
Price: 'actual_close'
Symbols: SP500 2012
Starting cash: $50,000
Start date: 1 January 2008
End date: 31 December 2009

When an event occurs, buy 100 shares of the equity on that day.
Sell 100 shares 5 trading days later.

The final value of the portfolio using the sample file is -- 2009,12,28,54824.0

Details of the Performance of the portfolio

Data Range :  2008-01-03 16:00:00  to  2009-12-28 16:00:00

Sharpe Ratio of Fund : 0.527865227084
Sharpe Ratio of $SPX : -0.184202673931

Total Return of Fund :  1.09648
Total Return of $SPX : 0.779305674563

Standard Deviation of Fund :  0.0060854156452
Standard Deviation of $SPX : 0.022004631521

Average Daily Return of Fund :  0.000202354576186
Average Daily Return of $SPX : -0.000255334653467
"""

# before processing/simulating the buy/sell file:
# put a $CASH, Sell, 50000 at the first date of the sim
# put a $CASH, Buy, 0 at the last date of the sim 

#$ events.py trade data/orders-buy-low.csv
#$ sim.py trade 50000 data/orders-buy-low.csv data/values-buy-low-fast.csv
#$ sim.py analyze data/values-buy-low-fast.csv



