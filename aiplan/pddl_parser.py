"""
pyparsing parser definition to parse STRIPS and PDDL files for AI Planning class (coursera)
"""
from traceback import print_exc

import pyparsing as pp

MAX_NUM_ARGS = 1000000000  # max of 1 billion arguments for any function (relation constant)

# function constants are usually lowercase, that's not a firm requirement in the spec
identifier = pp.Word( pp.alphas, pp.alphanums + "-_" )
variable   = pp.Word('?').suppress() + pp.Word(pp.alphas, pp.alphanums + '_')
comment    = pp.OneOrMore(pp.Word(';').suppress()) + pp.restOfLine('comment')
typ        = pp.Literal('-').suppress() + pp.Optional(pp.Literal(' ').suppress()) + identifier

state = pp.OneOrMore(pp.Literal('(').suppress() + pp.Group(pp.OneOrMore(identifier)) + pp.Literal(')').suppress())
state_conjunction = (pp.Literal('(') + pp.Keyword('and')).suppress() +  state + pp.Literal(')').suppress()

define       = pp.Keyword('define')   # (define (domain random-domain) ... or (define (problem random-pbl1) ...
domain       = pp.Keyword('domain')   # (define (domain random-domain) ... 
problem      = pp.Keyword('problem')  # (define (problem random-pbl1) ...
header       = define | domain | problem

init       = pp.Literal(':').suppress() + pp.Keyword('init')         # (:requirements :strips)
goal       = pp.Literal(':').suppress() + pp.Keyword('goal')        # (:requirements :typing)

state_name = pp.Literal('(').suppress() + (init | goal)
state_value = (state_conjunction | state)  + pp.Literal(')').suppress() # |
                  # pp.dictOf((init | goal), state))
# FIXME: this runs forever!
named_states = pp.dictOf(state_name, state_value)
s = '(:init\n     (S B B) (S C B) (S A C)\n     (R B B) (R C B))'
s = r'''(:init
        (S B B) (S C B) (S A C)
        (R B B) (R C B))
     (:goal (and (S A A)))'''
print('Input strips string:')
print(s)
parsed_states = named_states.parseString(s)
print('parsed init state:')
print(parsed_states.asDict())
# print('parsed goal state:')
# print(parsed_states.goal.asList())

requirements = pp.Keyword(':requirements')  # (:requirements :strips)
strips       = pp.Keyword(':strips')        # (:requirements :strips)
typing       = pp.Keyword(':typing')        # (:requirements :typing)
action       = pp.Keyword(':action')       # (:action op1 ...
parameters   = pp.Keyword(':parameters')   # :parameters (?x1 ?x2 ?x3)
precondition = pp.Keyword(':precondition') # :precondition (and (S ?x1 ?x2) (R ?x3 ?x1)) 
effect       = pp.Keyword(':effect')       # :effect (and (S ?x2 ?x1) (S ?x1 ?x3) (not (R ?x3 ?x1))))
keyword      = requirements | strips | typing | parameters | precondition | effect | init | goal

# keyword    = pp.Literal(":").suppress() + identifier

conjunction  = pp.Keyword('and')  # :precondition (and (S ?x3 ?x1) (R ?x2 ?x2))
disjunction  = pp.Keyword('or')   # :precondition (or (S ?x3 ?x1) (R ?x2 ?x2))
operator     = conjunction | disjunction
# operator = pp.Word('~&|')  # not, and, or

# PPL keywords ("Relation Constants")


# role = pp.Keyword('role')  # role(p) means that p is a player name/side in the game.
# inpt = pp.Keyword('input') # input(t) means that t is a base proposition in the game.
# base = pp.Keyword('base')  # base(a) means that a is an action in the game, the outcome of a turn.
# init = pp.Keyword('init')  # init(p) means that the datum p is true in the initial state of the game.
# next = pp.Keyword('next')  # next(p) means that the datum p is true in the next state of the game.
# does = pp.Keyword('does')  # does(r, a) means that player r performs action a in the current state.
# legal = pp.Keyword('legal')  # legal(r, a) means it is legal for r to play a in the current state.
# goal = pp.Keyword('goal')  # goal(r, n) means that player the current state has utility n for player r. n must be an integer from 0 through 100.
# terminal = pp.Keyword('terminal')  # terminal(d) means that if the datam d is true, the game has ended and no player actions are legal.
# distinct = pp.Keyword('distinct')  # distinct(x, y) means that the values of x and y are different.
# true = pp.Keyword('true')  # true(p) means that the datum p is true in the current state.

# # GDL-II Relation Constants
# sees = pp.Keyword('sees')  # The predicate sees(?r,?p) means that role ?r perceives ?p in the next game state.
# random = pp.Keyword('random')  # A predefined player that choses legal moves randomly

# # GDL-I and GDL-II Relation Constants
# relation_constant = role | inpt | base | init | next | does | legal | goal | terminal | distinct | true | sees | random

# # TODO: DRY this up
# # functions (keywords that should be followed by the number of arguments indicated)
# RELATION_CONSTANTS =  {
#     'role': 1, 'input': 2, 'base': 1, 'init': 1, 'next': 1, 'does': 2, 'legal': 2, 'goal': 2, 'terminal': 1,  'distinct': 2, 'true': 1,
#     'sees': 1, 'random': 1,
#     '<=': MAX_NUM_ARGS,
#     '&': 1,
#     }

# other tokens/terms
# identifier = pp.Word(pp.alphas + '_', pp.alphas + pp.nums + '_')
# # Numerical contant
# # FIXME: too permissive -- accepts 10 numbers, "00", "01", ... "09"
# number = (pp.Keyword('100') | pp.Word(pp.nums, min=1, max=2))
# # the only binary operator (relationship constant?)
# implies = pp.Keyword('<=')
# token = (implies | variable | relation_constant | number | pp.Word(pp.alphas + pp.nums))

# Define recursive grammar for nested paretheticals
# grammar = pp.Forward()
# expression = pp.OneOrMore(implies | variable | relation_constant | number | operator | identifier)
# nested_parentheses = pp.nestedExpr('(', ')', content=grammar) 
# grammar << (implies | variable | relation_constant | number | operator | identifier | nested_parentheses)
# sentence = (expression | grammar) + (comment | pp.lineEnd.suppress() | pp.stringEnd.suppress())
# game_description = pp.OneOrMore(comment | sentence)


enclosed     = pp.Forward()
nested_parens = pp.nestedExpr('(', ')', content=enclosed)
enclosed << (keyword | (action + identifier) | comment | nested_parens)
# expression = pp.OneOrMore(implies | variable | relation_constant | number | operator | identifier)
# sentence = (expression | grammar) + (comment | pp.lineEnd.suppress() | pp.stringEnd.suppress())
# domain_description = pp.OneOrMore(comment | expression)

def test():
    parsed_domain = enclosed.parseFile('random_domain.strips')
    parsed_problem = enclosed.parseFile('random_pbl1.strips')
    return parsed_domain, parsed_problem

def sandbox():
    """Based on http://stackoverflow.com/a/4802004/623735"""
    loose_grammar = pp.Forward()
    nestedParens = pp.nestedExpr('(', ')', content=loose_grammar) 
    loose_grammar << (
                 pp.OneOrMore(pp.Optional(':').suppress() + pp.Word(pp.alphanums + '-_')) 
               | pp.OneOrMore(pp.Optional('?').suppress() + pp.Word(pp.alphanums + '-_')) 
               | initial
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
