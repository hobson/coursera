# exam2

from itertools import permutations
from social import *

pref1 = [[A,B,C,D],[B,C,D,A],[C,B,D,A],[D,C,A,B]]

for lie in permutations([A,B,C,D]):
    print lie, [(chr(-k1), v1) for (v1, k1) in sorted(((v, -ord(k)) for (k, v) in \
                borda([list(lie)]+pref1[1:])[1].iteritems()), reverse=True)]



