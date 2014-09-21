#!/usr/bin/env python
'''tools for parsing and evaluating General Game Playing Game Description Language (GDL) expressions
GDL is a language defined at Berkeley for their GGP competitions
'''

import re

class gdl(object):
    '''
    >>> parse('start m23 white description 10 10')

    >>>
    '''

    def __init__(self, start=None, stop=None, play=None, abort=None):
        self.context = {
            'nil': None,
            'noop': 'noop',
            'start': start or self.start,
            'stop': stop or self.stop,
            'play': play or self.play,
            'abort': abort or self.abort,
        }

    def start(game_id, moves):
        pass

    def stop(game_id, moves):
        pass

    def abort(game_id, moves):
        pass

    def play(game_id, moves):
        pass

    def parse_line(self, s):
        words = s.split()
        expr = words[0] + '(' + ','.join(words[1:]) + ')'
        return expr
    
    def parse(self, s):
        return [self.parse_line(line) for line in s.split('\n')]

class gdl_re(object):
    NIL = re.compile(r'\bnil\b')
    NOOP = re.compile(r'\bnoop\b')

def eval_expr(expr):
    '''Transform a GDL expression into python code.'''
    return eval(expr, context=gdl)
