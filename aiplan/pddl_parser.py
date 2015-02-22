"""
pyparsing parser definition to parse STRIPS and PDDL files for AI Planning class (coursera)
"""
from traceback import print_exc

from pyparsing import Optional, Keyword, Literal, Word
from pyparsing import Combine, Group, OneOrMore, restOfLine, dictOf
from pyparsing import alphas, alphanums
from pyparsing import nestedExpr, Forward

MAX_NUM_ARGS = 1000000000  # max of 1 billion arguments for any function (relation constant)

# function constants are usually lowercase, that's not a firm requirement in the spec
identifier = Word( alphas, alphanums + "-_" )
variable   = Combine(Literal('?') + Word(alphas, alphanums + '_'))
comment    = Optional(OneOrMore(Word(';').suppress()) + restOfLine('comment')).suppress()
# typ        = Literal('-').suppress() + Optional(Literal(' ').suppress()) + identifier

# All mean the same thing: ground predicate, ground atom, ground_literal
# Any formula whose arguments are all ground terms (literals = non-variables)
ground_predicate = Literal('(').suppress() + Group(OneOrMore(identifier)) + Literal(')').suppress() + comment
arguments = sequence_of_variables = Literal('(').suppress() + Group(OneOrMore(variable)) + Literal(')').suppress()
# Norvig/Russel tend to call this a "fluent"
predicate        = Literal('(').suppress() + Group(identifier + OneOrMore(variable)) + Literal(')').suppress()
notted_predicate = Literal('(').suppress() + Keyword('not') + predicate + Literal(')').suppress()

# a set of ground atoms/predicates is a state, they are all presumed to be ANDed together (conjunction)
state_conjunction_implicit = OneOrMore(ground_predicate)
state_conjunction_explicit = (Literal('(') + Keyword('and')).suppress() + state_conjunction_implicit + Literal(')').suppress()
state = state_conjunction_explicit | state_conjunction_implicit

function_arguments  = Literal('(').suppress() +  Group(OneOrMore(variable)) + Literal(')').suppress()
expr         = Literal('(').suppress() + Group(identifier + OneOrMore(variable)) + Literal(')').suppress()
notted_expr  = Literal('(').suppress() + Keyword('not') + expr + Literal(')').suppress()
expr_set     =  Literal('(').suppress() + OneOrMore(expr) + Literal(')').suppress()

init       = Literal(':').suppress() + Keyword('init')         # (:requirements :strips)
goal       = Literal(':').suppress() + Keyword('goal')        # (:requirements :typing)

init_goal_states = dictOf(Literal('(').suppress() + (init | goal),
                             state + Literal(')').suppress() + comment ) 
s = r'''(:init
        (S B B) (S C B) (S A C)
        (R B B) (R C B))
     (:goal (and (S A A)))'''
print('Input strips string:')
print(s)
parsed_states = init_goal_states.parseString(s)
print('parsed init state:')
print(parsed_states.asDict())

problem_name = (Literal('(') + Keyword('problem')).suppress() + identifier + Literal(')').suppress() 
problem_domain =  (Literal('(') + Keyword(':domain')).suppress() + identifier + Literal(')').suppress() 
problem = comment + ( comment +
           (Literal('(') + Keyword('define')).suppress() + problem_name 
           + problem_domain
           + init_goal_states
           + Literal(')').suppress()
           ) + comment

# print('parsed goal state:')
# print(parsed_states.goal.asList())
precondition       = Literal('(').suppress() + Keyword('and').suppress() + OneOrMore(predicate) + Literal(')').suppress()  # :precondition (and (S ?x1 ?x2) (R ?x3 ?x1)) 
effect             = Literal('(').suppress() + Keyword('and').suppress() + OneOrMore(predicate | notted_predicate) + Literal(')').suppress()     # :effect (and (S ?x2 ?x1) (S ?x1 ?x3) (not (R ?x3 ?x1))))
#keyword            = requirements + strips_req | typing_req | parameters | precondition | effect | init | goal
actions  = dictOf((Literal('(') + Keyword(':action')).suppress() + identifier,
                dictOf(Keyword(':parameters') | Keyword(':precondition') | Keyword(':effect'),
                         (arguments | precondition | effect) ) + Literal(')').suppress()
           )  
#           + Literal(')'))    # (:action op1 ...

s =  '''(:action op1
           :parameters (?x1 ?x2 ?x3)
           :precondition (and (S ?x1 ?x2) (R ?x3 ?x1))
           :effect (and (S ?x2 ?x1) (S ?x1 ?x3) (not (R ?x3 ?x1))))
     '''
print(actions.parseString(s))

domain_name =    (Literal('(') + Keyword('domain') ).suppress() + identifier + Literal(')').suppress() 
domain_requirements = OneOrMore(   Literal('(').suppress()
                                    # dictOf( doesn't work like I'd like here
                                    + (Keyword(':requirements')  # (:requirements :strips)
                                       + OneOrMore(Literal(':').suppress() + (Keyword('strips')  | Keyword('typing'))))
                                    + Literal(')').suppress()
                                  )     # (:requirements :strips)
domain =    dictOf((Literal('(') + Keyword('define')).suppress() + domain_name,
                domain_requirements
                + actions
             + Literal(')').suppress()
            )
s = '''(define (domain random-domain)
        (:requirements :strips)
  (:action op1
    :parameters (?x1 ?x2 ?x3)
    :precondition (and (S ?x1 ?x2) (R ?x3 ?x1))
    :effect (and (S ?x2 ?x1) (S ?x1 ?x3) (not (R ?x3 ?x1))))
)
    '''
s = '''            (define (domain random-domain)
              (:requirements :strips)
              (:action op1
                :parameters (?x1 ?x2 ?x3)
                :precondition (and (S ?x1 ?x2) (R ?x3 ?x1))
                :effect (and (S ?x2 ?x1) (S ?x1 ?x3) (not (R ?x3 ?x1))))
              (:action op2
                :parameters (?x1 ?x2 ?x3)
                :precondition (and (S ?x3 ?x1) (R ?x2 ?x2))
                :effect (and (S ?x1 ?x3) (not (S ?x3 ?x1))))
)
'''
print('Input STRIPS string:')
print(s)
parsed_domain = domain.parseString(s)
print(parsed_domain)

# state_label = Literal('(').suppress() + (init | goal)
# expr_type  = Literal('(').suppress() + (parameters | precondition | effect)
# variable_state = (notted_expr | expr)  + Literal(')').suppress() # |
                  # dictOf((init | goal), state))



s = '''(define (domain random-domain)
  (:requirements :strips)
  (:action op1
    :parameters (?x1 ?x2 ?x3)
    :precondition (and (S ?x1 ?x2) (R ?x3 ?x1))
    :effect (and (S ?x2 ?x1) (S ?x1 ?x3) (not (R ?x3 ?x1))))
  (:action op2
    :parameters (?x1 ?x2 ?x3)
    :precondition (and (S ?x3 ?x1) (R ?x2 ?x2))
    :effect (and (S ?x1 ?x3) (not (S ?x3 ?x1)))))
'''

# wrapping a conjunction (x | y) with a `OneOrMore()` and an Optional(comment) within x or y causes infinite recursion
grammar = domain | problem | (domain + problem)

def test(path_or_str=None):
    import os

    s = '''
            (define (domain random-domain)
              (:requirements :strips)
              (:action op1
                :parameters (?x1 ?x2 ?x3)
                :precondition (and (S ?x1 ?x2) (R ?x3 ?x1))
                :effect (and (S ?x2 ?x1) (S ?x1 ?x3) (not (R ?x3 ?x1))))
              (:action op2
                :parameters (?x1 ?x2 ?x3)
                :precondition (and (S ?x3 ?x1) (R ?x2 ?x2))
                :effect (and (S ?x1 ?x3) (not (S ?x3 ?x1)))))

            (define (problem random-pbl1)
                (:domain random-domain)
                  (:init
                    (S B B) (S C B) (S A C)
                    (R B B) (R C B))
                  (:goal (and (S A A))))

            '''
    ans = grammar.parseString(s)
    assert(ans.asDict()['random-domain'].asDict()['op1'].asDict()[':effect'].asList() == [
        ['S', '?x2', '?x1'], ['S', '?x1', '?x3'], 'not', ['R', '?x3', '?x1']])
    if not path_or_str:
        return ans

    if isinstance(path_or_str, basestring):
        if os.path.isfile(path_or_str):
            print('Parsing STRIPS file at: ' + path_or_str)
            s = open(path_or_str, 'r').read()
            if os.path.isfile(s):
                raise ValueError('path_or_str must not be a path to a file that contains another valid path!\n' + path_or_str )
            return test(s)  # a file whos contents is a path to itself
        else:
            print('Parsing STRIPS PDDL string: ' + path_or_str)
            return grammar.parseString(path_or_str)
    elif isinstance(path_or_str, (list, tuple)):
        return [test(p) for p in path_or_str]
    return False

if __name__ is '__main__':
    print(test())


def sandbox():
    """Based on http://stackoverflow.com/a/4802004/623735"""
    loose_grammar = Forward()
    nestedParens = nestedExpr('(', ')', content=loose_grammar) 
    loose_grammar << (
                 OneOrMore(Optional(':').suppress() + Word(alphanums + '-_')) 
               | OneOrMore(Optional('?').suppress() + Word(alphanums + '-_')) 
               | init
               | goal
               | ',' 
               | nestedParens)
    examples = [
        # definitely not PDDL-compliant, but parser does OK anyway (not strict)
        '(some global things (:a (nested list of three varibles (?list0 ?list1 ?list2))))',
        # this is a valid line of STRIPS (subset of PDDL grammar?)
        '(:requirements :strips)',
        # another valid line of STRIPS (subset of PDDL grammar?)
        '(define (domain random-domain))',
        # a complete (if simple) STRIPS problem definition from coursera AI Planning class, HW wk2
        r'''
            (define (problem random-pbl1)
              (:domain random-domain)
              (:init
                 (S B B) (S C B) (S A C)
                 (R B B) (R C B))
              (:goal (and (S A A))))
        ''',
        # a complete STRIPS domain definition from coursera AI Planning class, HW wk2
        r'''
        (define (domain random-domain)
          (:requirements :strips)
          (:action op1
            :parameters (?x1 ?x2 ?x3)
            :precondition (and (S ?x1 ?x2) (R ?x3 ?x1))
            :effect (and (S ?x2 ?x1) (S ?x1 ?x3) (not (R ?x3 ?x1))))
          (:action op2
            :parameters (?x1 ?x2 ?x3)
            :precondition (and (S ?x3 ?x1) (R ?x2 ?x2))
            :effect (and (S ?x1 ?x3) (not (S ?x3 ?x1)))))
        ''',
        ]
    ans = []
    for ex in examples:
        try:
            ans += [loose_grammar.parseString(ex).asList()]
            print(ans[-1])
        except:
            print_exc()
    return ans


