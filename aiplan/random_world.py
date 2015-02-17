"""
STRIPS planner uses simplified (no event) PDDL based on predicate first order logic which has relations, objects, and types of objects.

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
"""

State = set()

def proposition_kwargs(proposition):
    """Return a dict of positional arguments for a proposition's parameters"""
    return dict(enumerate(proposition[1:]))

class RandomWorld(object):
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
    operators =  {
        'op1': {
                'precondition': {('S', 0, 1): True, ('R', 2, 0): True},
                'effect': {('S', 1, 0): True, ('S', 0, 2): True, ('R', 2, 0): False},
            },
        'op2': {'precondition': {('S', 2, 0): True, ('R', 1, 1): True},
                'effect': {('S', 0, 2): True, ('S', 2, 0): False}},
        }
    operators =  {
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

    def __init__(self, initial=None):
        self.state = State()
        if initial:
            self.state |= initial
        self.applicable_actions = set()

    def preconditions(self, name):
        """ground/evaluate preconditions with args substituted into the appropriate slots in an operation

        Returns:
            2-tuple: (negative_preconditions, positive_preconditions)
                This allows you to e.g. preconditions()[True] to get positive preconditions
        """
        return self.evaluate_action(name, 'precondition')

    def evaluate_preconditions(self, name, args):
        """ground/evaluate preconditions with args substituted into the appropriate slots in an operation

        Returns:
            2-tuple: (negative_preconditions, positive_preconditions)
                This allows you to e.g. preconditions()[True] to get positive preconditions
        """
        return self.evaluate_action(name, 'precondition', args)

    def effects(self, name, args):
        """ground/evaluate effects by substituting args into the appropriate slots in an operation
        
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
            args = tuple(range(len(self.operators[name][evaluate_effect])))

        for k, is_positive in self.operators[name][evaluate_effect].iteritems():
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

    def add_applicable_actions(self, action_name, applicable_actions=set(), remaining_preconditions=None, action_args=tuple()):
        """
        preconditions = (positive_preconditions, negative_preconditions)
        args = list of arguments to the operator

        Dock-Worker-Robot example move action:
            1. precondition = ('adjacent', 'from', 'to')
            2. 

        Returns:
            set of applicable instances for a given operator (`action_name`) in the current state
        """
        if remaining_preconditions is None:
            remaining_preconditions = self.preconditions(action_name, action_args)
        if not remaining_preconditions[True]:
            for neg_precon in remaining_preconditions[False]:
                if neg_precon in self.state:
                    return
            applicable_actions.add()
                # if self.falsifies(self.state)
        else:
            # check positive preconditions
            for pos_precon in remaining_preconditions[True]:
                # look for all the state propositions in the current state that might be able to match this positive precondition
                for state_proposition in self.state:
                    # if the predicates don't match you can skip it
                    if state_proposition[0] != pos_precon[0]:
                        continue
                    # is it OK to add substitutions in place or do we need to copy and extend them?
                    # extend the substitution such that the state_proposition and the positive_preconditions match
                    # a substitution is a dict of variables and the values that they should take on
                    new_kwargs = dict(act_kwargs)   # | set([pos_precon[1:]])
                    new_kwargs = update_act_kwargs(new_kwargs, proposition_kwargs(state_proposition))
                    # check to see if the new substituion is a valid action in the current state, if it is add it to applicable_actions
                    if new_kwargs:
                        new_preconditions = set(preconditions).discard(pos_precon)
                        self.add_applicable_actions(action_name, applicable_actions, new_preconditions, new_kwargs)


class RandomProblem1(RandomWorld):
    """(define (problem random-pbl1)
      (:domain random-domain)
      (:init
         (S B B) (S C B) (S A C)
         (R B B) (R C B))
      (:goal (and (S A A))))
    """
    goal = State()

    def __init__(self):
        initial = State()
        initial |= set([('S', 'B', 'B'), ('S', 'C', 'B'), ('S', 'A', 'C'),
                        ('R', 'B', 'B'), ('R', 'C', 'B')])
        super(RandomProblem1, self).__init__(initial=initial)
        self.goal = State()
        self.goal |= set([('S', 'A', 'A')])

    def goal_test(self):
        # TODO: can probably be simplified into a single all(self.state.get(k, False) == bool(v) for k, v ...)
        return (all(self.state.get(k, None) for k, v in self.goal.iteritems() if v)
                and not any(self.state.get(k, None)) for k, v in self.goal.iteritems() if not v)


