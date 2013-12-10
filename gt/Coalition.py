from itertools import combinations, permutations
from collections import OrderedDict as od
from collections import Mapping
import math
import numpy as np

import decorators as decorate


class Coalition:

    def __init__(self, values=None, N=None, verbosity=1):
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
            2. Name players
        """
        self.verbosity = verbosity
        self.v = None
        self.N = None
        if isinstance(values, Mapping):
            self.v = od(values)
        else:
            N_subsets = len(values)
            N0 = inverse_num_subsets(N_subsets, includes_empty_set=True)
            N1 = inverse_num_subsets(N_subsets, includes_empty_set=False)
            if N0 != N and N1 != N:
                if int(N0) == N0:
                    N = N0
                else:
                    N = int(N1)
            # TODO: build function to compute N from the length of the list of values
            if isinstance(N, int):
                lists_of_index_tuples = [list(combinations(range(N), i)) for i in range(1, N + 1)]
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
                self.N = N
        if self.v and not self.N:
            self.N = max(max(index_tuple) for index_tuple in self.v) + 1
        # self.N = invserse_stirling(len())?

    def shapley_values(self):
        """
        The list of values for each player that they contribute to a coalition or society.

        3-agent example:
        phi1 = 1/3. * v1 + 1/6. * (v12-v2) + 1/6. * (v13-v3) + 1/3. * (v123-v23)
        phi2 = 1/3. * v2 + 1/6. * (v12-v1) + 1/6. * (v23-v3) + 1/3. * (v123-v13)
        phi3 = 1/3. * v3 + 1/6. * (v13-v1) + 1/6. * (v23-v2) + 1/3. * (v123-v12)
        And this gives the doctest answer below as:
        phi1 = 1/3 + 10/3 + 100/3 = 222/6 = 37
        phi2 = 2/3 + 31/6 + 110/3 = 255/6 = 42.5
        phi3 = 3/3 + 33/6 + 111/3 = 261/6 = 43.5

        >>> Coalition([1, 2, 4], N=2).shapley_values()
        [1.5, 2.5]
        >>> Coalition([1, 2, 3, 12, 13, 23, 123], N=3).shapley_values()
        [37.0, 42.5, 43.5]
        >>> Coalition([0, 0, 0, 3, 3, 0, 4], N=3).shapley_values()  # doctest: +ELLIPSES
        [2.333..., 0.833..., 0.833...]

        TODO:
            reduce redundant use of tuple(sorted()) to sort tuples and convert back to tuples
        """
        ans = [0.] * self.N
        # because we're doing all the permutations individually, we don't need to normalize by neither
        #   1. the number of ways we can add players and reach that marginal contribution of a player
        #   nor
        #   2. the number of ways we could add to that subset to acheive the grand coalition
        for player in range(self.N):
            if self.verbosity:
                print '----- Player %d -----' % player
            for order_added in permutations(range(self.N)):
                when_added = order_added.index(player)
                player_subset = tuple(sorted(order_added[:(when_added + 1)]))
                # initial value doesn't care when this player was added, just that she was added
                value = self.v[player_subset]
                # if any other players were added before this one, then need to subtrace the value they contributed
                if when_added > 0:
                    value -= self.v[tuple(sorted(order_added[:when_added]))]
                if self.verbosity:
                    print 'Marginal value for subset %s is %s' % (player_subset, value)
                # weight by the number of possible ways we could have added players before this one
                weight1 = 1.  # math.factorial(when_added)
                #print '%s ways we could have added players before %s' % (weight1, player)
                # weight by the number of possible ways additional players could be added to complete the society of size self.N
                weight2 = 1.  # math.factorial(self.N - when_added - 1)
                if self.verbosity > 1:
                    print '%s ways we could added players after %s' % (weight2, player)
                    print 'total weight = %s, total value = %s ' % (weight1 * weight2, value * weight1 * weight2)
                ans[player] += (value * weight1 * weight2)
        # normalize by the number of possible permutations in a society of self.N "players"
        for i in range(len(ans)):
            ans[i] = ans[i] / float(math.factorial(self.N))
        return ans

    def core(self):
        """
        Calculate one possible core allocation, or None, if the core is empty.

        >>> Coalition([0, 0, 0, .8, .8, .8, 1], verbosity=0).core()
        >>> Coalition([0, 0, 0, 2/3., 2/3., 2/3., 1], verbosity=0).core()  # doctest: +ELLIPSES
        [0.333..., 0.333..., 0.333...]
        >>> Coalition([0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 2], N=4, verbosity=0).core()
        """
        # need to solve N! inequalities, so might as well solve the equality first
        dim = len(self.v) - self.N
        A = np.zeros((dim, self.N))
        y = self.v.values()[self.N:]
        #print len(y)
        for i, indices in enumerate(list(self.v)[self.N:]):
            for j in indices:
                A[i][j] = 1.
        if self.verbosity > 2:
            print 'A * x = y'
            print 'A = %s' % A
            print 'y = %s' % y
        #print np.linalg.solve(A, y)
        ans, residuals, rank, singular_values = np.linalg.lstsq(A, y)
        if self.verbosity:
            print 'Possible core allocation = %s' % ans
        bigger = np.squeeze(np.asarray(A * np.matrix([[a] for a in ans])))
        if self.verbosity > 1:
            print 'y for solution = %s' % bigger
        # Subtract 1e-12 to deal with potential round off error when comparing floats,
        #   especially when using approx lstsq() instead of exact solve()
        #   I haven't had a solution abandonded due to roundoff error yet, but wanted to make sure
        if any((b < (y[i] - 1e-12 * (1 + abs(b) + abs(y[i])))) for (i, b) in enumerate(bigger)):
            if self.verbosity > 1:
                print 'The potential core values on the left are smaller than the society value allocations on the right.'
                print [(b, (y[i] - 1e-12 * (1 + abs(b) + abs(y[i])))) for (i, b) in enumerate(bigger)]
            return None
        indie_values = self.v.values()[:self.N]
        if any(a < (indie_values[i] - 1e-12 * (1 + abs(a) + abs(indie_values[i]))) for i, a in enumerate(ans)):
            if self.verbosity > 1:
                print 'The potential core values on the left are smaller than the individual player value allocations on the right.'
                print [(a, (indie_values[i] - 1e-12 * (1 + abs(a) + abs(indie_values[i])))) for i, a in enumerate(ans)]
            return None
        return list(ans)

    def __repr__(self):
        return 'Coalition(%s, N=%s)' % (self.v, self.N)


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


def num_subsets(N, include_empty_set=False):
    """
    Number of nonempty subsets that you can form from N distinguishable elements.

    >>> num_subsets(4)
    15
    >>> num_subsets(4, include_empty_set=True)
    16
    """
    return 2 ** N - int(not(include_empty_set))


def inverse_num_subsets(N_subsets, includes_empty_set=False):
    """
    Number of nonempty subsets that you can form from N distinguishable elements.

    >>> inverse_num_subsets(15)
    4
    >>> num_subsets(16, include_empty_set=True)
    4
    """
    return math.log(N_subsets + int(not(includes_empty_set)), 2)


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
