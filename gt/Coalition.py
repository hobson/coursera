from itertools import combinations as combos
from collections import OrderedDict as od
from collections import Mapping
import math
import numpy as np

import decorators as decorate


class Coalition:

    def __init__(self, values=None, N=None):
        """The only argument is a list of N! lists of values for players and subsets of coalitions

        [[v1],
         [v2],
         :
         [vN],
         [v12, v13, ..., V1N],
         [v21, v23, ..., V2N],
         :
         [vN1, vN2, ..., VNN],
         [v123, v124, ..., V12N],
         :
         [v132, v134, ..., V13N],
         [v213, v214, ..., V21N],
         :
         [v142, v143, ..., V14N],
         :

        TODO:
           1. Better representation of all the possible coalitions and subsets of the coalitions
              with less redundancy, perhaps as a hypercube rather than a list of matrices
        """
        self.v = None
        if isinstance(values, Mapping):
            self.v = od(values)
        else:
            # TODO: build function to compute N from the length of the list of values
            if isinstance(N, int):
                lists_of_index_tuples = [list(combos(range(N), i)) for i in range(1, N + 1)]
                index_tuples = [item for sublist in lists_of_index_tuples for item in sublist]
                self.v = od((i, None) for i in index_tuples)
                if values and isinstance(values, (list, tuple)):
                    if isinstance(values[0], (int, float)):
                        pass
                    else:
                        # flatten a shallow list of lists
                        values = [item for sublist in values for item in sublist]
                    for index_tuple, value in zip(self.v, values):
                        self.v[index_tuple] = value

    def shapley_values(self):
        return None

    def __repr__(self):
        return self.v


# http://extr3metech.wordpress.com/2013/01/21/stirling-number-generation-using-python-code-examples/
def stirling(n, k, cache=None):
    """
    >>> stirling(1, 1)
    1
    >>> stirling(3, 3)
    1
    >>> stirling(3, 2)
    3
    >>> stirling(3, 1)
    2
    # These are all incorrect but intended according to the http://extr3metech.wordpress.com/ blog
    >>> stirling(4, 2)
    11  # FIXME: mproperly gives 7 as the answer
    >>> stirling(5, 2)
    15
    >>> stirling(5, 3)
    25
    >>> stirling(20,15)
    452329200
    """
    cache = cache or {(0, 0): -1}
    if n <= 0:
        return 1
    elif k <= 0:
        return 0
    elif n == k:
        return 1
    elif n < k:
        return 0
    elif (n, k) in cache:
        return cache[(n, k)]
    else:
        cache[(n - 1, k - 1)] = stirling(n - 1, k - 1, cache)
        cache[(n - 1, k)] = stirling(n - 1, k, cache)
        return (k * cache[(n - 1, k)]) + cache[(n - 1, k - 1)]


#http://en.wikipedia.org/wiki/Stirling_numbers_of_the_first_kind#Table_of_values_for_small_n_and_k
# N \  k  0   1   2   3   4   5   6   7   8   9
#
# 0       1
# 1       0   1
# 2       0   1   1
# 3       0   2   3   1
# 4       0   6   11  6   1
# 5       0   24  50  35  10  1
# 6       0   120 274 225 85  15  1
# 7       0   720 1764    1624    735 175 21  1
# 8       0   5040    13068   13132   6769    1960    322 28  1
# 9       0   40320   109584  118124  67284   22449   4536    546 36  1
unsigned_stirling_type1_cache = \
    {
        (0, None): (1,),
        (1, None): (0, 1),
        (2, None): (0, 1, 1),
        (3, None): (0, 2, 3, 1),
        (4, None): (0, 6, 11, 6, 1),
        (5, None): (0, 24, 50, 35, 10, 1),
        (6, None): (0, 120, 274, 225, 85, 15, 1),
        (7, None): (0, 720, 1764, 1624, 735, 175, 21, 1),
        (8, None): (0, 5040, 13068, 13132, 6769, 1960, 322, 28, 1),
        (9, None): (0, 40320, 109584, 118124, 67284, 22449, 4536, 546, 36, 1),
    }


@decorate.memoize
def unsigned_stirling_type1(N, k=None):
    """
    >>> unsigned_stirling_type1(9, 1)
    40320
    >>> unsigned_stirling_type1(0, 0)
    1
    >>> unsigned_stirling_type1(1234, 0)
    0
    >>> unsigned_stirling_type1(1000, 1000)
    1
    >>> unsigned_stirling_type1(1000, 1001)
    """
    polymul = np.polymul
    # for large N, use pure python convolution to avoid numpy integer overflow
    if N > 19:
        polymul = convolve
        #raise ValueError("Integer overflow is likely for numpy's 64-bit integers.")
    if k is None:
        if N == 0:
            return (1,)
        return tuple(int(value) for value in polymul([1, N], unsigned_stirling_type1(N - 1, None)))
    if N >= 0:
        if N == k:
            return 1
        if k == 0:
            return 0
        if k > N:
            return None  # blank in http://en.wikipedia.org/wiki/Stirling_numbers_of_the_first_kind#Table_of_values_for_small_n_and_k
        return unsigned_stirling_type1(N - 1, None)[-k]
    return None


def unsigned_stirling_type1_table(N=12):
    """
    Compute a lower triangular matrix (table) of unsigned Stirling numbers of the first kind

    May take longer than a minute to run for N > 1000, as RAM is exhausted.

    >>> sum([1*any(x < 0 for x in row) for row in unsigned_stirling_type1_table(200)])
    0
    """
    table = []
    for n in range(N + 1):
        row = []
        for k in range(n + 1):
            row += [unsigned_stirling_type1(n, k)]
        table += [row]
    return table


def convolve(a, b):
    """
    Generate the discrete, linear convolution of two one-dimensional sequences.

    based on Gareth Reese answer at http://stackoverflow.com/a/12129472/623735
    """
    return [sum(a[j] * b[i - j] for j in range(i + 1)
                if j < len(a) and i - j < len(b))
            for i in range(len(a) + len(b) - 1)]


@decorate.memoize
def signed_stirling_type1(n, k):
    """Returns the stirling number of the first kind using recursion.."""
    if n == 0 and k == 0:
        return 1
    if k == 0 and n >= 1:
        return 0
    if k > n:
        return 0
    return signed_stirling_type1(n - 1, k - 1) - (n - 1) * signed_stirling_type1(n - 1, k)


@decorate.memoize
def signed_stirling_type2(n, k):
    """Returns the stirling number Stirl2(n,k) of the second kind using recursion."""
    if k <= 1 or k == n:
        return 1L
    if k > n or n <= 0:
        return 0L
    return signed_stirling_type2(n - 1, k - 1) + k * signed_stirling_type2(n - 1, k)


# see http://mathforum.org/kb/message.jspa?messageID=342551&tstart=0
# and http://www.physicsforums.com/showthread.php?t=205761&highlight=inverse+factorial+gamma
def lambert_w(x):
    """
    >>> lambert_w(5 * math.exp(5))
    5.0
    """
    try:
        L1 = math.log(x)
        L2 = math.log(L1)
    except:
        raise ValueError('%s does not have an integer Lambert_W value.' % x)
        return None
    # accurate to only 4 digits
    return (L1 - L2 + L2 / L1
            + L2 * (-2. + L2) / (2. * L1 ** 2.)
            + L2 * (6 - 9 * L2 + 2 * L2 ** 2) / (6. * L1 ** 3)
            + L2 * (-12 + 36 * L2 - 22 * L2 ** 2 + 3 * L2 ** 3) / (12. * L1 ** 4.))


def lambert_l(x):
    # digamma(x) = scipy.special.digamma(x) = d/dx (gamma(x)) / math.gamma(x) = d/dx (math.log(gamma(x))
    k = 1.461632  # approximately (the digamma zero or root)
    c = math.sqrt(2 * math.pi) / math.e - math.gamma(k)
    #c = 0.036534  # approximate
    return math.log((x + c) / math.sqrt(2 * math.pi))


def approximate_inverse_gamma(x):
    return lambert_l(x) / lambert_w(lambert_l(x) / math.e) + 0.5


def inverse_factorial(x):
    """
    >>> inverse_factorial(6)
    3
    >>> inverse_factorial(2)
    2
    >>> inverse_factorial(math.factorial(17))
    17
    >>> inverse_factorial(math.factorial(100))
    100
    >>> inverse_factorial(math.factorial(170))
    170
    >>> inverse_factorial(math.factorial(169))
    169
    >>> sum(inverse_factorial(math.factorial(i)) != i for i in range(2, 171))
    0
    """
    if x < 2:
        raise ValueError("This function is invalid for x < 2. " +
                         "Two values of x in math.factorial(x) give a value of 1, and no values of x will ever give a value less than 1.")
    if x in inverse_factorial.lookup_table:
        return inverse_factorial.lookup_table.index(x)
    if x > 1e307:  # 77257415615307998967396728211129263114716991681296451376543577798900561843401706157852350749242617459511490991237838520776666022565442753025328900773207510902400430280058295603966612599658257104398558294257568966313439612262571094946806711205568880457193340212661452800000000000000000000000000000000000000000L:
        raise ValueError("Unable to calculate the inverse factorial for large numbers (> 1e307)")
    try:
        ans = int(round(approximate_inverse_gamma(x)) - 1)
        if math.factorial(ans) == x:
            return ans
    except:
        raise ValueError('%s does not have a natural number (or integer) inverse factorial.' % x)
inverse_factorial.lookup_table = [1, 1, 2, 6, 24, 120, 720]
