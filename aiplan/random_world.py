"""
STRIPS planner uses simplified PDDL based on predicate first order logic which has relations, objects, and types of objects.
PDDL adds to the STRIPS syntax events and negatation of fluents/relations in states

predicate = relation
"?" prepended to variable names
" - " separates a variable and its type
":" prepended to names of relations


dock worker robot domain example:
    # topology of the domain (static, stay the same for a given state)
    adjacent(l1: location, l2: location)
    attached(p: pile, l: location)
    belong(k: crane, l: location)

    # fluent relations (can change from state to state)
    at(r: robot, l: location)
    occupied(r: robot, c; container)

    loaded(r: robot, c: container)
    unloaded(r: robot)

    holding(k: crane, c: continer)
    empty(k: crane)

    in(c: container, p: pile)
    top(c: container, p: pile)
    on(c: container, c: container)

an atom is a predictate and all it's arguments/parameters
a ground atom is a predicate with fixed (not variable) arguments/parameters
a state is a set of ground atoms
an atom "holds" in a state if it is among the set of ground atoms in that state

random world domain:
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

random problem in pddl:
    (define (problem random-pbl1)
      (:domain random-domain)
      (:init
         (S B B) (S C B) (S A C)
         (R B B) (R C B))
      (:goal (and (S A A))))

defines two actions
1. op1(x1, x2. x3):

2. op2()

Examples:
    >>> Problem('domain1_problem1.strips')  # doctest: +NORMALIZE_WHITESPACE
    <Problem(goal=set([('S', 'A', 'A')]), state=State([('R', 'B', 'B'), ('R', 'C', 'B'), ('S', 'C', 'B'), ('S', 'B', 'B'), ('S', 'A', 'C')]))>
      for <World(actions={'op1': {'negative_effects': set([('R', 'x3', 'x1')]), 'positive_preconditions': set([('R', 'x3', 'x1'), ('S', 'x1', 'x2')]), 'positive_effects': set([('S', 'x2', 'x1'), ('S', 'x1', 'x3')]), 'parameters': ('x1', 'x2', 'x3'), 'negative_preconditions': set([])}, 'op2': {'negative_effects': set([('S', 'x3', 'x1')]), 'positive_preconditions': set([('R', 'x2', 'x2'), ('S', 'x3', 'x1')]), 'positive_effects': set([('S', 'x1', 'x3')]), 'parameters': ('x1', 'x2', 'x3'), 'negative_preconditions': set([])}})>

"""
from pyparsing import ParseResults
from pddl import grammar
import os


class State(set):
    pass

def proposition_kwargs(proposition):
    """Return a dict of positional arguments for a proposition's parameters"""
    return dict(enumerate(proposition[1:]))

def extend_kwargs(kwargs, more_kwargs):
    extended_kwargs = dict(kwargs)
    if any(more_kwargs.get(k, v) != v for k, v in kwargs.iteritems()):
        return False
    extended_kwargs.update(more_kwargs)
    return extended_kwargs

class World(object):

    def __init__(self, state=None, actions=None, verbosity=None):
        self.verbosity = verbosity if verbosity is not None else self.verbosity

        self.actions = {}   # perhaps there should be Domain class separate from the World class 
        if actions:
            self.actions = dict(actions)

        self.state = State(state or State())

        self.applicable_actions = set()  # this belongs in the World or State class (not the Domain class)
        self.add_applicable_actions()

    def preconditions(self, name):
        """ground/evaluate preconditions with args substituted into the appropriate slots in an action

        Returns:
            2-tuple: (negative_preconditions, positive_preconditions)
                This allows you to e.g. preconditions()[True] to get positive preconditions
        """
        return self.evaluate_action(name, 'precondition')

    def evaluate_preconditions(self, name, args):
        """ground/evaluate preconditions with args substituted into the appropriate slots in an action
        Returns:
            2-tuple: (negative_preconditions, positive_preconditions)
                This allows you to e.g. preconditions()[True] to get positive preconditions
        """
        return self.evaluate_action(name, 'precondition', args)

    def effects(self, name, args):
        """ground/evaluate effects by substituting args into the appropriate slots in an action
        
        Returns:
            2-tuple: (negative_effects, positive_effects) or (delete_list, add_list)
                This allows you to do, e.g., `effects(...)[True]` to get the "add-list"
        """
        return self.evaluate_action(name, 'effect', args)

    def evaluate_action(self, name, evaluate_effect, args=None):
        """

        Args:
            action_name (str): PDDL action name (STRIPS calls this the "operator name")
            evaluate_effect (bool or int): 0/False = eval. preconditions, 1/True = eval. effect
            action_args (tuple or list): ordered list of arguments to substitute into the action's parameter variables
        
        Returns:
            2-tuple: (negative, positive) propositions
        """
        negative, positive = set(), set()

        if args is None:
            args = tuple(range(len(self.actions[name][evaluate_effect])))

        for k, is_positive in self.actions[name][evaluate_effect].iteritems():
            evaluated_proposition = tuple([k[0]] + [args[k[i]] for i in range(1, len(k))])
            if is_positive:
                positive.add(evaluated_proposition)
            else:
                negative.add(evaluated_proposition)
        return negative, positive

    def is_valid_action(self, name, args):
        preconditions = self.preconditions(name, args)
        return (all(pp in self.state for pp in preconditions[True]) and
                not any(np in self.state for np in preconditions[False]))

    def act(self, name, *args):
        # actions = {
        # 'op1': {
        #         'precondition': {('S', args[0], args[1]): True, ('R', args[2], args[0]): True},
        #         'effect': {('S', args[1], args[0]): True, ('S', args[0], args[2]): True, ('R', args[2], args[0]): False},
        #     },
        # 'op2': {'precondition': {('S', args[2], args[0]): True, ('R', args[1], args[1]): True},
        #         'effect': {('S', args[0], args[2]): True, ('S', args[0], args[2]): False}},
        # }
        if self.is_valid_action(name, args):
            effects = self.effects(name, args)
            # delete all negated relations first
            for effect in effects[False]:
                self.state.discard(effect)
            # then insert all positive relations
            for effect in effects[True]:
                self.state.add(effect)
            return True
        return False  # invalid action attempted

    def update_act_kwargs(act_kwargs, new_kwargs):
        for k, v in new_kwargs.iteritems():
            if k in act_kwargs:
                if act_kwargs[k] != v:
                    return None
            else:
                act_kwargs[k] = v
        return act_kwargs

    def add_applicable_actions(self, action_names=None, applicable_actions=None, remaining_preconditions=None, action_kwargs=dict()):
        """
        preconditions = (positive_preconditions, negative_preconditions)
        args = list of arguments to the operator

        Dock-Worker-Robot example add_applicable_action('move',...) using 11-line algorithm:
            2.  check len of positive preconditions list (nonempty so go to else at 6 then 7)
            7.  first positive precondition = ('adjacent', 'from', 'to')
            8.  first "adjacent" state proposition = ('adjacent', 'loc1', 'loc2')
            9.  new_substitutions = sigma_prime =  set([('adjacent', ('from', 'loc1'), ('to', 'loc2'))])
            10. ('from', 'loc1') and ('to', 'loc2') is_valid = True because they're the first/only substitutions
            11. recursive call to add_applicable_actions, now with nonempty substitutions/sigma set
            7.  check next positive_precondition = ('at', 'r', 'from')
            8.  restart loop over states and check first "at" state = ('at', 'r1', 'loc2')  
            9.  new_substitutions = sigma_prime =  set([('adjacent', ('from', 'loc1'), ('to', 'loc2')),
                                                       ('at', ('r', 'r1'), ('from', 'loc2'))
                                                      ])
            10. ('from', 'loc2') is invalid because contradicts earlier substitution ('from', 'loc1')
            8.  proceed to next state proposition "adjacent" in for loop = ('adjacent', 'loc2', 'loc1')
            9.  new_substitutions = sigma_prime =  set([('adjacent', ('from', 'loc2'), ('to', 'loc1'))])
            10. ('from', 'loc1') and ('to': 'loc1') is_valid because it's the only substitution (nothing to contradict)
            11. recursive call add_applicable_actions, now with nonempty substitutions/sigma set

        Returns:
            set of applicable instances for a given operator (`action_name`) in the current state
        """
        if applicable_actions is None:
            self.applicable_actions = set()
            applicable_actions = self.applicable_actions
        if action_names is None:
            action_names = tuple(self.actions.keys())
            remaining_preconditions = None
        elif isinstance(action_names, basestring):
            action_name = action_names  #action_names = tuple(an.strip() for an in action_names.split(' ') if an.strip())
            # # First call to add_applicable_actions should populate the preconditions set
            if remaining_preconditions is None:
                remaining_preconditions = self.actions[action_name]['negative_preconditions'], self.actions[action_name]['positive_preconditions']
                if self.verbosity:
                    print('Initial set of preconditions is {0}'.format(remaining_preconditions))
            # 2. check if positive preconditions still left
            if not remaining_preconditions[True]:
                # 3. for every negative precondition...
                for neg_precon in remaining_preconditions[False]:
                    # 4. if state falsifies the tuple(action, action_kwargs)...
                    if neg_precon in self.state: return
                # 5. A.add((action_name, kwargs))
                applicable_actions.add((action_name, tuple(action_kwargs.items())))
                    # if self.falsifies(self.state)
            # 6.
            else:
                # 7. choose a positive precondition (TODO: consider using `for` loop but this is used recursively)
                pos_precon = iter(remaining_preconditions[True]).next()  # .pop() would delete it before we can be sure that's OK
                if self.verbosity:
                    print('using positive_precondition {0}'.format(pos_precon))            # 8. look for all the state propositions in the current state that might be able to match this positive precondition
                for state_proposition in self.state:
                    if self.verbosity:
                        print('Using state proposition {0}'.format(state_proposition))            # 8. look for all the state propositions in the current state that might be able to match this positive precondition
                    # 8. if the predicates don't match you can skip it
                    if state_proposition[0] != pos_precon[0]:
                        continue
                    # FIXME: is it OK to add substitutions in place or do we need to copy and extend them as this has?
                    # extend the substitution such that the state_proposition and the positive_preconditions match
                    # a substitution is a dict of variables and the values that they should take on
                    # 9. add substitution: sigma_prime = sigma.extend()
                    new_kwargs = extend_kwargs(action_kwargs, dict(zip(pos_precon[1:], state_proposition[1:])))
                    if self.verbosity:
                        print('sigma_prime = {0}'.format(new_kwargs))
                    # 10. check to see if the new substituion is a valid action in the current state, if it is add it to applicable_actions
                    if new_kwargs:
                        new_preconditions = tuple(rp.copy() for rp in remaining_preconditions)
                        new_preconditions[True].remove(pos_precon)
                        self.add_applicable_actions(action_name, applicable_actions, new_preconditions, new_kwargs)
            return
        self.applicable_actions = set()
        for action_name in action_names:
            self.add_applicable_actions(action_name, self.applicable_actions)

class RandomWorld(World):
    """(define (domain random-domain)
      (:requirements :strips)
      (:action op1
        :parameters (?x1 ?x2 ?x3)
        :precondition (and (S ?x1 ?x2) (R ?x3 ?x1))
        :effect (and (S ?x2 ?x1) (S ?x1 ?x3) (not (R ?x3 ?x1))))  ; FIXME: Does "not" mean to negate the existing state or just set it to False?
      (:action op2
        :parameters (?x1 ?x2 ?x3)
        :precondition (and (S ?x3 ?x1) (R ?x2 ?x2))
        :effect (and (S ?x1 ?x3) (not (S ?x3 ?x1)))))  ; FIXME: Does "not" mean to negate the existing state or just set it to False?
    """
    verbosity = 1
    actions =  {
        'op1': {
            'arguments': ('x1', 'x2', 'x3'),
            'positive_preconditions': set([('S', 'x1', 'x2'), ('R', 'x3', 'x1')]),
            'negative_preconditions': set(),
            'positive_effects': set([('S', 'x2', 'x1'), ('S', 'x1', 'x3')]),  # add-list
            'negative_effects': set([('R', 'x3', 'x1')]),                     # delete-list
               },
        'op2': {
            'arguments': ('x1', 'x2', 'x3'),
            'positive_preconditions': set([('S', 'x3', 'x1'), ('R', 'x2', 'x2')]),
            'negative_preconditions': set(),
            'positive_effects': set([('S', 'x1', 'x3')]),                     # add-list
            'negative_effects': set([('S', 'x3', 'x1')]),                     # delete-list
               },
        }

    def __init__(self, state=None, actions=None, verbosity=None):
        # because these are defined as class vars
        if actions is not None:
            self.actions = dict(actions)
        return super(RandomWorld, self).__init__(state=state, actions=self.actions, verbosity=verbosity)


class Problem(World):
    goal = State()
    initial = State()

    def __init__(self, state=None, actions=None, goal=None, verbosity=1):
        self.actions = {}
        if isinstance(state, basestring):
            if os.path.isfile(state):
                self.parse_file(state)
            else:
                self.parse_str(state)
            state = None
        super(Problem, self).__init__(state=state or self.state, actions=actions or self.actions, verbosity=verbosity)
        self.initial = self.state
        self.goal = goal or self.goal or State()


    def ingest_parse_results(self, parse_results=None):
        self.parse_results = parse_results or self.parse_results or ParseResults()
        self.initial =  set(tuple(flu) for flu in self.parse_results.get('init', self.initial))
        self.state = self.initial

        # FIXME: parser sometimes returns a list of lists and other times returns a list of strings (if there's only one fluent in the goal state!)
        # FIXME: DRY this up and do the same thing for the initial state in case the parser hoses it up for a single fluent too!
        self.goal =     [flu if isinstance(flu, basestring) else tuple(flu) for flu in self.parse_results.get('goal', self.goal)]
        if isinstance(self.goal[0], tuple):
            self.goal = set(self.goal)
        else: 
            self.goal = set([tuple(self.goal)])
        self.domain_name = iter(k for k in parse_results.keys() if k not in ('init', 'goal')).next()
        actions = dict(((action_name, dict(((k2, tuple(v3 if isinstance(v3, basestring) else tuple(v3) for v3 in v2.asList())) 
                            for k2, v2 in action_dict.iteritems()))) 
                                for action_name, action_dict in self.parse_results[self.domain_name].iteritems()))
        for name, action in actions.iteritems():
            self.actions[name] = self.actions.get(name, {})            
            self.actions[name]['parameters'] = tuple(param[1:] if param.startswith('?') else param for param in action['parameters'])
            
            # FIXME: DRY up this repeated code for postprocessing preconditions and effects
            self.actions[name]['positive_preconditions'], self.actions[name]['negative_preconditions'] = set(), set()
            polarity = 'positive'
            for flu in action['precondition']:
                if flu is 'not':
                    polarity = 'negative'
                    continue
                self.actions[name][polarity + '_preconditions'] |= set([tuple(x[1:] if x.startswith('?') else x for x in flu)])
                polarity = 'positive'

            self.actions[name]['positive_effects'], self.actions[name]['negative_effects'] = set(), set()
            polarity = 'positive'
            for flu in action['effect']:
                if flu is 'not':
                    polarity = 'negative'
                    continue
                self.actions[name][polarity + '_effects'] |= set([tuple(x[1:] if x.startswith('?') else x for x in flu)])
                polarity = 'positive'

    def parse_file(self, path):
        self.parse_results = grammar.parseFile(path)
        self.ingest_parse_results(self.parse_results)

    def parse_str(self, s):
        self.parse_results = grammar.parseFile(s)
        self.ingest_parse_results(self.parse_results)

    def goal_test(self):
        # TODO: can probably be simplified into a single all(self.state.get(k, False) == bool(v) for k, v ...)
        return (all(self.state.get(k, None) for k, v in self.goal.iteritems() if v)
                and not any(self.state.get(k, None)) for k, v in self.goal.iteritems() if not v)

    def __repr__(self):
        return '<Problem(goal={0}, state={1})>\n  for <World(actions={2})>'.format(self.goal, self.state, self.actions)


class RandomProblem1(Problem):
    """(define (problem random-pbl1)
      (:domain random-domain)
      (:init
         (S B B) (S C B) (S A C)
         (R B B) (R C B))
      (:goal (and (S A A))))
    """
    verbosity = 1
    goal = State([('S', 'A', 'A')])
    initial = State([('S', 'B', 'B'), ('S', 'C', 'B'), ('S', 'A', 'C'),
                                     ('R', 'B', 'B'), ('R', 'C', 'B')])
    actions =  {
        'op1': {
            'positive_preconditions': set([('S', 'x1', 'x2'), ('R', 'x3', 'x1')]),
            'negative_preconditions': set(),
            'positive_effects': set([('S', 'x2', 'x1'), ('S', 'x1', 'x3')]),  # add-list
            'negative_effects': set([('R', 'x3', 'x1')]),                     # delete-list
               },
        'op2': {
            'positive_preconditions': set([('S', 'x3', 'x1'), ('R', 'x2', 'x2')]),
            'negative_preconditions': set(),
            'positive_effects': set([('S', 'x1', 'x3')]),                     # add-list
            'negative_effects': set([('S', 'x3', 'x1')]),                     # delete-list
               },
        }

    def __init__(self, state=None, actions=None, goal=None, verbosity=None):
        # because these are defined as class vars
        # if actions is not None:
        #     self.actions = dict(actions)
        return super(RandomProblem1, self).__init__(state=state or self.initial, actions=actions or self.actions, goal=goal or self.goal, verbosity=verbosity)

