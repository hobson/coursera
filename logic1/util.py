#!/usr/bin/env python
# -*- coding: latin-1 -*-
"""Logic utitilities

truth_table: Mapping of variable logic values to formula evaluation
"""

from sympy import sympify
#from sympy.logic.boolalg import BooleanFunction
from itertools import product


def truth_table(formula):
    """Mapping of variable logic values to formula evaluation

    >>> truth_table(u'∼(∼r∨r))')
    """
    mathjax_trans = {u'∼': 'not ', u'~': 'not ', u'∨': '|'}
    if isinstance(formula, basestring):
        for math_chr, sym_chr in mathjax_trans.iteritems():
            formula = formula.replace(math_chr, sym_chr)
        formula = sympify(formula)
    symbols = sorted(formula.free_symbols)

    return [(sym_values, +formula.subs(dict(zip(symbols, sym_values)))) 
            for sym_values in product([0, 1], repeat=len(symbols))]
