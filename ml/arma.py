import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime as dt
from scipy.optimize import leastsq

sales = np.loadtxt('sales.csv', delimiter=',')
returns = np.loadtxt('returns.csv', delimiter=',')
calls = np.loadtxt('calls.csv', delimiter=',')

for t, x in [sales, returns, calls]
doy = np.array([dt.fromordinal(int(o)).timetuple().tm_yday for o in t])
mon = np.array([dt.fromordinal(int(o)).timetuple().tm_ymon for o in t])
yr = np.array([dt.fromordinal(int(o)).year for o in t])

# reserve 10% of data as test sample
cutoff = 0.9 * len(x)

# could just as easily use range(1,13)
months = sorted(set(mon))
avgs = np.zeros(13)

# for each month of the year
for i in months:
    # get the indices for that month (but save some data)
    indices = np.where(mon[:cutoff] == i)
    avgs[i-1] = x[indices].mean()


def subtract_avgs(a, doy):
   return a - avgs[doy.astype(int)-1]
 
def subtract_trend(a, poly, b):
   return a - poly[0] * b - poly[1]
 
def print_stats(a):
   print "Min", a.min(), "Max", a.max(), "Mean", a.mean(), "Std", a.std()
   print
 

without_avgs = subtract_avgs(x[:cutoff], doy[:cutoff])
print "After Subtracting DOY avgs"
print_stats(without_avgs)
 
# Step 2. Linear trend
trend = np.polyfit(yr[:cutoff], less_avgs, 1)
print "Trend coeff", trend
less_trend = subtract_trend(less_avgs, trend, yr[:cutoff])
print "After Subtracting Linear Trend"
print_stats(less_trend)
 
def model(prediction, lag2, lag1):
   l1, l2 = prediction
 
   return l2 * lag2 + l1 * lag1
 
def error(prediction, actual, lag2, lag1):
   return actual - model(prediction, lag2, lag1) 
 
p0 = [1.06517683, -0.08293789]
params = leastsq(error, p0, args=(less_trend[2:], less_trend[:-2], less_trend[1:-1]))[0]
print "AR params", params
 
#Step 1. again
less_avgs = subtract_avgs(x[cutoff+1:], doy[cutoff+1:])
 
#Step 2. again
less_trend = subtract_trend(less_avgs, trend, yr[cutoff+1:])
 
delta = np.abs(error(params, less_trend[2:], less_trend[:-2], less_trend[1:-1]))
print "% delta less than 2", (100. * len(delta[delta <= 2]))/len(delta)
 
plt.hist(delta, bins = 10, normed = True)
plt.show()