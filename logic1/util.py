#!/usr/bin/env python
# -*- coding: latin-1 -*-
"""Logic utitilities

truth_table: Mapping of variable logic values to formula evaluation
"""

import sympy
from sympy import sympify
# from sympy.logic.boolalg import BooleanFunction
from itertools import product
#from tabulate import tabulate


def make_expr(expr):
    r'''Ensure string or math expression is a valid sympy.Expr object

    >>> make_expr('a>>b')
    Expr(Implies(a, b))
    >>> make_expr('(~((~p&q)|r)>>s)')
    Expr(Implies(And(Not(r), Or(Not(q), p)), s))
    >>> make_expr('(∼((∼p&q)∨r)⊃s)')
    Expr(Implies(And(Not(r), Or(Not(q), p)), s))
    '''
    #expr = unicode(expr)
    #mathjax_trans = {u'∼': 'not ', u'~': 'not ', u'∨': '|'}
    mathjax_symbols = [('∼', '~'), ('∨', '|'), ('⊃', '>>'), (',', '&'), ('≡', '='), ('', '&')]  # , u'\u223c': '~', u'\u2228': '|'}
    numerical_logic = []  # {'~': '~+', '&': '&+', '|': '|+'}  # , u'\u223c': '~', u'\u2228': '|'}
    for math_chr, sym_chr in mathjax_symbols:
        expr = expr.replace(math_chr, sym_chr)
    for bool_chr, sym_chr in numerical_logic:
        expr = expr.replace(bool_chr, sym_chr)
    return sympy.Expr(sympify(expr))


def truth_table(formula, typ=str, header=True):
    r'''Mapping of variable logic values to formula evaluation

    >>> truth_table('∼(∼r∨r)')  # doctest: +NORMALIZE_WHITESPACE
    [['r', '='], ['0', '0'], ['1', '0']]
    >>> truth_table('∼(∼r∨r)', typ=int)  # doctest: +NORMALIZE_WHITESPACE
    [['r', '='], [0, 0], [1, 0]]
    >>> truth_table('∼(((r⊃∼p)⊃p)⊃(q⊃p))', typ=int)  # doctest: +NORMALIZE_WHITESPACE
    [['p', 'q', 'r', '='],
     [0, 0, 0, 0],
     [0, 0, 1, 0],
     [0, 1, 0, 0],
     [0, 1, 1, 0],
     [1, 0, 0, 0],
     [1, 0, 1, 0],
     [1, 1, 0, 0],
     [1, 1, 1, 0]]
    >>> truth_table('∼((r∨(r∨∼r))⊃(p⊃(∼p⊃∼r)))', typ=int)  # doctest: +NORMALIZE_WHITESPACE
    [['p', 'r', '='], [0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0]]
    >>> truth_table('(∼(p⊃∼r)⊃(p⊃r))', typ=int)  # doctest: +NORMALIZE_WHITESPACE
    [['p', 'r', '='], [0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]]
    '''
    formula = make_expr(formula)
    symbols = sorted(formula.free_symbols)

    numerical_logic = [('Expr', ''), ('(', ''), (')', ''), ('True', '1'), ('False', '0')]
    table = []
    if header:
        table += [[str(sym) for sym in symbols] + ['=']]
    for sym_values in product([0, 1], repeat=len(symbols)):
        #print sym_values
        #print str(+formula.subs(dict(zip(symbols, sym_values))))
        ans = str(+formula.subs(dict(zip(symbols, sym_values))))
        for old, new in numerical_logic:
            ans = ans.replace(old, new)
        #print ans
        table += [list(typ(val) for val in sym_values) + [typ(int(ans))]]
    return table


def is_valid(argument):
    r'''Compute a truth table for an argument (proof) and return True if it is valid, otherwise False

    An argument (proof) is valid if and only if all branches of the proof tree
    have a contradiction. This is the same as checking all possible values of 
    the variables in a conjunction of the premises and the negation of the
    conclusion and never finding a way to make the overall conjunction True.
    This is the same as being unable to make the conclusing False when all the 
    premises are True.

    >>> is_valid('∼((r&p)&((q⊃r)&(∼r⊃∼r))),∼(((r∨p)∨r)∨∼q) therefore (∼(∼r∨p)⊃((r∨∼p)∨(p⊃∼p)))')
    True
    >>> is_valid('∼(∼(p⊃∼r)⊃p),∼(p∨(p&q)) therefore ∼(q∨∼p)')
    True
    >>> is_valid('(∼(∼q∨∼p)∨p),∼((r⊃p)⊃(p∨(p&p))) therefore ∼(∼(r⊃∼q)∨r)')
    True
    >>> is_valid('∼(∼p&∼q),∼(((q⊃∼r)∨(r∨p))&r) therefore ∼(∼(q⊃r)∨r)')
    False
    '''
    return not any(bool(row[-1]) for row in truth_table(argument.replace('therefore', '&~'), typ=bool, header=False))


tautologies = { 
    r'∼∼A ≡ A': 'double negation',
    r'(A∨ ∼A) BOE ∼(A  ∼A)': 'excluded middle',
    r'((A  B)  C) ≡ (A  (B  C))': 'associativity of &',
    r'((A∨B)∨C) ≡ (A∨(B∨C))': 'associativity of ∨',
    r'(A  B) ≡ (B  A)': 'commutativity of &',
    r'(A∨B) ≡ (B∨A)': 'commutativity of ∨',
    r'(A  (B∨C)) ≡ ((A  B)∨(A  C))': 'distribution of & over ∨',
    r'(A∨(B  C)) ≡ ((A∨B)  (A∨C))': 'distribution of ∨ over &',
    r'∼(A  B) ≡ (∼A∨ ∼B)': "De Morgan's rule for ~ of & to ∨",
    r'∼(A∨B) ≡ (∼A  ∼B)': "De Morgan's rule for ~ of ∨ to &",
    r'∼(∼A  ∼B) ≡ (A∨B)': "De Morgan's rule for ~ of & of ~ to ∨",
    r'∼(∼A∨ ∼B) ≡ (A  B)': "De Morgan's rule for ~ of ∨ of ~ to &",
    r'(A ⊃ B) ≡ (∼A∨B)': 'material implication',
    r'∼(A ⊃ B) ≡ (A  ∼B)': 'negated conditional',
    r'(A  B) ⊃ A': 'conditional for &, 1st conjunct',
    r'(A  B) ⊃ B': 'conditional for &, 2nd conjunct',
    r'A ⊃ (A∨B)':  'conditional for ∨, 1st disjunct',
    r'B ⊃ (A∨B)': 'conditional for ∨, 2nd disjunct',
    r'A ⊃ (B ⊃ A)': 'weaking',
    r'(A ⊃ (B ⊃ C)) ⊃ ((A ⊃ B) ⊃ (A ⊃ C))': 'distribution for conditional',
    r'(∼A ⊃ ∼B) ⊃ (B ⊃ A)': 'contraposition 1',
    r'(A ⊃ B) ⊃ (∼B ⊃ ∼A)': 'contraposition 2',
    r'((A ⊃ B) ⊃ A) ⊃ A': "Pierce's Law",
    r'(A  (A ⊃ B)) ⊃ B': 'Modus ponens conditional',
    r'((A  B) ⊃ C) ≡ (A ⊃ (B ⊃ C))': "Curry's formula",
    r'((A∨B)  ∼A) ⊃ B': 'excluding disjuncts, 1st disjunct',
    r'((A∨B)  ∼B) ⊃ A': 'excluding disjuncts, 2nd disjunct',
}