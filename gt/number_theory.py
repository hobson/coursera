"""
Number theory and combinatorics

References
  ASPN Recipes, mathematics.py
  http://adorio-research.org/wordpress/?p=11460
"""

import gmpy
import xmath


class memoize:

    def __init__(self, f):
        self.d = {}
        self.f = f

    def __call__(self, *i):
        try:
            return self.d[i]
        except:
            r = apply(self.f, i)
            self.d[i] = r
            return r


def gcd(a, b):
    """
    Greatest common divisor of a and b.
    Uses the wonderful ancient algorithm of Euclid.
    """
    a = abs(a)
    b = abs(b)
    if a == 0:
        return b
    if b == 0:
        return a
    while b != 0:
        (a, b) = (b, a % b)
    return a


def agcd(nlist):
    """
    Greatest common divisor of numbers in nlist.
    """
    L = len(nlist)
    if len(nlist) == 0:
        return 0

    r = nlist[0]
    for i in range(1, L):
        r = gcd(r, nlist[i])
        if r <= 1:
            return r
    return r


def lcm(a, b):
    """
    Least common multiple of a and b.
    """
    if a == 0 or b == 0:
        return 0
    return (a * b) / gcd(a, b)


def alcm(nlist):
    """
    Least common multiple of numbers in nlist.
    """
    L = len(nlist)
    if len(nlist) == 0:
        return 0

    r = nlist[0]
    for i in range(1, L):
        r = lcm(r, nlist[i])
        if r <= 1:
            return r
    return r


def factorial(n):
    """
    Returns the factorial of n.
    n! = n * (n-1) * (n-2) * ... * 2 * 1

    Remarks:
       Joel Neely, ASPN Recipes has the crisp functional implementation of factorial:

    return reduce (lambda x,y: x* y, range(1, n+1))
    """
    # Version 1: iteration
    p = 1
    for i in range(2, n + 1):
        p *= i
    return p

    # Version 2: recursion
    if n <= 1:
        return 1
    return n * factorial(n - 1)


def risingFactorial(x, n):
    """
    Returns value of x (x + 1) (x + 2) ... (x+n-1)
    """
    p = x
    for i in range(1, n):
        p *= (x + i)
    return p


def fallingFactorial(x, n):
    """
    Returns value of x (x-1) (x-2) ... (x - n + 1)
    """
    p = x
    for i in range(1, n):
        p *= (x - i)
    return p


def nCr(n, r):
    """
    Computes the number of combinations of n things
    taken r at a time.
    """
    # Check boundaries. This may need to be revised.
    if r < 0 or n < 0:
        return 0
    if (n <= 0 or r == n or r == 0):
        return 1

    # Version 1: using Gmpy
    return int(gmpy.bincoef(n, r))

    # Version 2: using iteration.
    if r > n - r:
        r = n - r
    x = n
    i = n - 1
    for d in range(2, r + 1):
        x = (x * i) / d
        i = i - 1
    return x

    # Version 3: using recursion
    if n < 2 or r == 0 or r >= n:
        return 1
    return nCr(n - 1, r) + nCr(n - 1, r - 1)


def nPr(n, r):
    """ Non recursive version """
    x = 1
    for i in range(n-r + 1, n+1):
       x *= i
    return x
 
    """Returns the value of nPr = n! / [ (n -r)! ] using recursion. """
    if r == 0: return 0L
    if r == 1: return n
    return nPr(n, r - 1) * (n - r + 1)


#from math import exp, pi, sqrt

def nIntPartitions(j, k = 1):
    """
    Returns using recursion the number of integer partitions.
    """
    total = 1
    j -= k
    while (j>=k):
        total += nIntPartitions(j, k)
        j, k = j-1, k+1
    return total

    # """
    # Ramanujan's formula for
    # upper bound for number of partitions of k
    # """
    # return int(exp(pi*sqrt(2.0*n/3.0))/(4.0*n*sqrt(3.0)))
 
 
def ackermann(m, n):
    """Returns the ackermann function using recursion. (m, n are non-negative)."""
    if m == 0 and n >= 0: return n + 1
    if n == 0 and m >= 1: return ackermann(m-1, 1)
    return ackermann(m-1, ackermann(m, n-1))
 
def bell(n):
    """Returns the nth Bell number using recursion."""
    if n < 2: return 1
    sum = 0
    for k in range(1, n+1):
        sum = sum + nCr(n-1, k-1) * bell(k-1)
    return sum
 
def bernoulli(n):
    """
    Returns recursively the nth bernoulli number as a fraction.
    """
    if n <= 0: return 1
    if n == 1: return gmpy.mpq(-1,2)
    if (n%2) == 1:  return 0
    return -gmpy.mpq(sum(map(lambda x:gmpy.bincoef(n+1, x)*bernoulli(x), range(n))), n+1)
 
 
def catalan(n):
    """Returns the nth Catalan number using recursion."""
    if n <= 2: return n
    return (2*n)*(2*n-1) * catalan(n-1) / ((n +1)*n)
 
 
def catalan2(n):
    """Returns the nth Catalan number using a second slower (very very slooow) recursion."""
    if n < 2: return 1L
 
    sum = 0L
    for i in range(1, n):
        sum = sum + catalan2(i) * catalan2(n-i)
    return sum
 
 
def fibo(n):
    """
    nth Fibonacci number.
    See fibonacci how to implement this.
    """
    return gmpy.fib(n)
 
def fibonacci(n):
    """
    nth Fibonacci number.
    """
 
    # Version 1: iterative
    a = b = 1
    if n <= 2:
       return b
    for i in range(2,n+1):
       a, b  = b, a + b
    return b
 
    # Version 2: recursion
    if n < 2:
        return 1L
    return fibonacci(n-1) + fibonacci(n-2)
 
 
def lucas(n):
    """Returns the nth lucas number."""
    if n < 2:
        return 1
    if n == 2:
        return 3
    a = 1
    b = 3
    for n in range(3, n+1):
        c = a + b
        a, b = b,c
    return c
 
    # Recursive version
    return lucas(n-1) + lucas(n-2)
 
 
def stirling1(n,k):
    """Returns the stirling number of the first kind using recursion.."""
    if n == 0 and k == 0:
       return 1
    if k == 0 and n >= 1:
       return 0
    if k > n:
       return 0
    return stirling1(n-1, k-1) - (n - 1) * stirling1(n-1, k)
 
def stirling2(n,k):
    """Returns the stirling number Stirl2(n,k) of the second kind using recursion."""
    if k <= 1 or k == n: return 1L
    if k > n or n <= 0: return 0L
    return stirling2(n-1, k-1) + k * stirling2(n-1, k)
 
def nDerangements(n):
    """
    Returns the number of derangements using the formula:
        d(n) = n d(n-1) + (-1)^n
 
    A derangement is a permutation where each element is not in its
    natural place, or p[i] != i.
    """
    if n == 1:
        return 0
    d  = -1
    dn = 0
    for i in range(2, n+1):
        dnm1 = dn
        d    = -d
        dn   = i * dnm1 + d
    return dn
 
 
nIntPartitions = xmath.memoize(nIntPartitions)
 
ackermann = xmath.memoize(ackermann)
bell      = xmath.memoize(bell)
bernoulli = xmath.memoize(bernoulli)
catalan   = xmath.memoize(catalan)
catalan2  = xmath.memoize(catalan2)
lucas     = xmath.memoize(lucas)
stirling1 = xmath.memoize(stirling1)
stirling2 = xmath.memoize(stirling2)