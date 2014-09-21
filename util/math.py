#math

import numpy as np
import scipy as sci

def linear_correlation(x, y=None, ddof=0):
    """Pierson linear correlation coefficient (-1 <= plcf <= +1)
    >>> linear_correlation(range(5), [1.2 * x + 3.4 for x in range(5)])
    1.0
    >>> abs(linear_correlation(sci.rand(2, 1000)))  # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
    0.0...
    """
    if y is None:
        if len(x) == 2:
            y = x[1]
            x = x[0]
        elif len(x[0]) ==2:
            y = [yi for xi, yi in x] 
            x = [xi for xi, yi in x]
    return np.cov(x, y, ddof=ddof)[1,0] / np.std(x, ddof=ddof) / np.std(y, ddof=ddof)


def best_correlation_offset(x, y, ddof=0):
    """Find the delay between x and y that maximizes the correlation between them
    A negative delay means a negative-correlation between x and y was maximized
    """
    def offset_correlation(offset, x=x, y=y):
        N = len(x)
        if offset < 0:
            y = [-1 * yi for yi in y]
            offset = -1 * offset 
        # TODO use interpolation to allow noninteger offsets
        return linear_correlation([x[(i - int(offset)) % N] for i in range(N)], y)
    return sci.minimize(offset_correlation, 0)




