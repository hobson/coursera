import json
from gmpy2 import mpz, c_divmod, gcd, is_prime # ,mpq,mpfr,mpc
from gmpy2 import gcdext as egcd

# # prime
# p = mpz(13407807929942597099574024998205846127479365820592393377723561443721764030073546976801874298166903427690031858186486050853753882811946569946433649006084171)
# # not prime:
# g = mpz(11717829880366207009516117596335367088558084999998952205599979459063929499736583746670572176471460312928594829675428279466566527115212748467589894601965568)
# # not prime:
# h = mpz(3239475104050450443565264378728065788649097520952449527834792452971981976143292558073856937958553180532878928001494706097394108577585732452307673444020333)

B = mpz(2)**20
#Z = mpz(2)**40 + 1

# p = 101
# g = 63
# h = 53
# B = 2**2

# find x such that h = g**x in the set prime integers for g and h prime too
# h/g**x1 = g**(B*x0) 0 <= (x0 and x1) <= 2**20

from progressbar import Bar, ETA, Percentage, ProgressBar, ReverseBar, RotatingMarker, SimpleProgress, Timer


def compute_lookup(h, base=None, B=None, p=None):
    """
    >>> table = compute_lookup(h=53, base=63, B=2**2, p=101)
    >>> table == {81: 1, 59: 2, 53: 0, 39: 4, 33: 3}
    True
    """
    g = base
    x1hash = {h: 0}
    widgets = [Bar('*'), ' ', Percentage(), ' ',  Timer(), ' ',  ETA()]
    pbar = ProgressBar(widgets=widgets, maxval=B).start()
    invg = discrete_inverse(g, base=p)
    k = h
    for x1 in range(1, B+1):
        k = (k * invg) % p
        x1hash[int(k)] = x1
        #print x1, k
        if not x1 % 10000:
            #print i, x1, k
            # print len(x1hash)
            pbar.update(x1)
        x1hash[int(k)] = int(x1)
    pbar.finish()
    #print x1hash
    with open('x1hash.json', 'w') as fp:
        json.dump(x1hash, fp, indent=2)
    return x1hash


def discrete_log(h, base=None, B=None, m=None, lookup=None):
    """m = modulus of the discrete log
    >>> discrete_log(h=53, base=63, B=2**2, m=101)
    9
    >>> discrete_log(h=(456 ** 789) % 1123, base=456, B=2**10, m=1123)
    789
    >>> discrete_log(h=(456 ** 789) % 1123, base=456, B=2**15, m=1123)
    789
    """
    g = base
    if not lookup:
        lookup = compute_lookup(h=h, base=g, B=B, p=m)
    gb = (g ** B) % m
    x1 = 0
    for x0 in range(B+1):
        x0 = mpz(x0)
        if (gb ** x0) % m in lookup:
            x1 = lookup[((g ** B) ** x0) % m]
            break
    return int(x0 * B + x1)


def euclidean_algorithm(a,b):
    """The Euclidean algorithm for finding the greatest common divisor

    Equivalent to gmpy2.gcd(a, b) but slower for large integers
    """
    while a:
            a, b = b%a, a
    return b
#gcd = euclidean_algorithm


def extended_euclidian_algorithm(a, b):
    """Compute g, x, y such that a*x + b*y = g = gcd(a, b)

    Equivalent to gmpy2.gcdext(a, b) but slower for large integers

    GCD = greatest common divisor
    a, b positive integers.

    Boneh, Stanford 2014, Coursera Cryptography 1 reversed the use of the varables
        x <-> a and y <-> b for the inputs and outputs respectively.
    """
    x,y, u,v = 0,1, 1,0
    while a != 0:
        q, r = b//a, b%a
        m, n = x - u*q, y - v*q
        b,a, x,y, u,v = a,r, u,v, m,n
    return b, x, y
#egcd = extended_euclidian_algorithm


def discrete_inverse(x, base):
    """Equivalent to gmpy.invert(a=x, m=base)"""
    nonsingular_if_one_gcd, x_inv, y = extended_euclidian_algorithm(x, base)
    # if the gcd of x and base is not 1, then x is not invertible 
    if nonsingular_if_one_gcd == 1:
        return x_inv
#invert = inverse = discrete_inverse


def Z_star(N):
    """The invertible elements within Z, the set of integers 1,2,...N

    if gcd(x, N) == 1 then the element is invertible.

    >>> Z_star(12)
    [1, 5, 7, 11]
    >>> Z_star(35)[20:]
    [31, 32, 33, 34]
    """
    return [x for x in range(1, N) if discrete_inverse(x, N) is not None]
invertible_elements = Z_star


def ord(g, p):
    '''Smallest a>0 such that g**a % p = 1'''
    # if not is_prime(p):
    #     raise ValueError("The order of %r can only be computed for module prime number. %r is not prime" % (g, p))
    a = mpz(1)
    g = mpz(g)
    p = mpz(p)
    while ((g**a) % p != 1):
        a += 1
    return int(a)

def main():
    p = mpz(13407807929942597099574024998205846127479365820592393377723561443721764030073546976801874298166903427690031858186486050853753882811946569946433649006084171)
    # # not prime:
    g = mpz(11717829880366207009516117596335367088558084999998952205599979459063929499736583746670572176471460312928594829675428279466566527115212748467589894601965568)
    # # not prime:
    h = mpz(3239475104050450443565264378728065788649097520952449527834792452971981976143292558073856937958553180532878928001494706097394108577585732452307673444020333)
    
    discrete_log(h=h, base=g, B=2**20, m=p)
