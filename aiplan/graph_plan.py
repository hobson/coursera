from __future__ import print_function
import six
import future
import warnings
from queue import PriorityQueue as BuiltinPriorityQueue
import math
import random
from traceback import print_exc
from aima.search import Node, PriorityQueue
from pug.decorators import force_hashable

import networkx as nx
import re

from nltk.corpus import wordnet as wn

# NORVIG AIMA code
infinity = float('inf')


"""
FIXME:
    - prob(verbosity=0) doesn't take

Examples:
    >>> node = astar_search(NPuzzleProblem(initial=[1,6,4,8,7,0,3,2,5], verbosity=0))                      
    >>> len(node.path())
    23 or 38
    >>> node.depth
    37
    >>> astar_search(NPuzzleProblem(initial=[8,1,7,4,5,6,2,0,3], verbosity=0)).depth
    81
    >>> # Find the number of unique nodes/states at a given depth (distance) from the goal
    >>> PDDL parser for planning domains and problems then test code on this problem:
    (define (problem random-pb1)
      (:domain random-domain)
      (:init
       (S B B) (S C B) (S A C)
       (R B B) (R C B)
      )
     (:goal 
      (and (S A A))
     )
    )
    >>> # clearly suboptimal, because getting different answer eacy time
    >>> min(astar_search(NPuzzleProblem(initial=[1,6,4,8,7,0,3,2,5], verbosity=0)).depth for i in range(10))
    21
    >>> min(astar_search(NPuzzleProblem(initial=[8,1,7,4,5,6,2,0,3], verbosity=0)).depth for i in range(10))
    25
    >>> len(nodes_at_depth(NPuzzleProblem(initial=range(9), verbosity=0), depth=27, verbosity=0))
    6274
"""

class Problem(object):
    """The abstract class for a graph search problem.
    Should override the methods `.actions()`, and `.result()`
    If result(action) == action then no need to override (this is default `result()`).
    May override `.__init__()`, `.goal_test()`, and `.path_cost()`.
    """
    def __init__(self, initial, goal=None, verbosity=1):  # , graph=nx.Graph()):
        """The constructor specifies the initial state, and possibly a goal
        state, if there is a unique goal.  Your subclass's constructor can add
        other arguments."""
        self.initial, self.goal, self.verbosity = initial, goal, verbosity

    def actions(self, state):
        """Return the actions that can be executed in the given
        state. The result would typically be a list, but if there are
        many actions, consider yielding them one at a time in an
        iterator, rather than building them all at once."""
        pass

    def result(self, state, action):
        """Return the state that results from executing the given
        action in the given state. The action must be one of
        self.actions(state)."""
        return action

    def goal_test(self, state):
        """Return True if the state is a goal. The default method compares the
        state to self.goal, as specified in the constructor. Override this
        method if checking against a single self.goal is not enough."""
        if self.verbosity:
            print('{0} =? {1}'.format(state, self.goal))
        state = getattr(state, 'state', state)
        return force_hashable(state) == force_hashable(self.goal)

    def path_cost(self, c, state1, action, state2):
        """Return the cost of a solution path that arrives at state2 from
        state1 via action, assuming cost c to get up to state1. If the problem
        is such that the path doesn't matter, this function will only look at
        state2.  If the path does matter, it will consider c and maybe state1
        and action. The default method costs 1 for every step in the path."""
        return c + 1

    def value(self, state):
        """For optimization problems, each state has a value.  Hill-climbing
        and related algorithms try to maximize this value."""
        pass

    def pretty_str(self, state):
        return str(state)

def swap(seq, i0=1, i1=0):
    """Swap elements in a sequence

    Maintains the type of sequence (useful for swapping elements in an immutable tuple),
    which means it won't swap anything for `set`s, and `dict`s  result a list of keys.
    Operates on mutable sequences (e.g. `list`s) in-place.
    For iterables 

    >>> swap([1, 2, 3])
    [2, 1, 3]
    >>> swap(xrange(5), 2)
    [2, 1, 0, 3, 4]
    >>> swap(tuple(xrange(5)), 3, 4)
    (0, 1, 2, 4, 3)
    >>> swap(range(5), 1, 4)
    [0, 4, 2, 3, 1]
    >>> swap(set([1,2,3]))
    [2, 1, 3]
    """
    typ = None
    if not hasattr(seq, '__setitem__') or not not hasattr(seq, '__getitem__'):
        # set-like objects don't have an order, so should remain a list when done 
        if not hasattr(seq, '__and__'):
            typ = type(seq)
        seq = list(seq)
    seq[i0], seq[i1] = seq[i1], seq[i0]
    try:
        return typ(seq)
    except TypeError:
        return seq


def shuffled(seq):
    """Shuffle elements in a sequence

    Maintains the type of sequence (useful for shuffling elements in an immutable tuple)

    >>> set(shuffled([1, 2, 3]))
    {1, 2, 3}
    >>> len(shuffled(range(5)))
    5
    """
    typ = None
    if not hasattr(seq, '__setitem__') or not not hasattr(seq, '__getitem__'):
        # set-like objects don't have an order, so should remain a list when done 
        if not hasattr(seq, '__and__'):
            typ = type(seq)
        seq = list(seq)
    random.shuffle(seq)
    try:
        return typ(seq)
    except TypeError:
        return seq


class NPuzzleProblem(Problem):
    """States are a sequence of the digits 0-9

    0 represents a blank. 
    So the actions available are to move a nonzero tile adjacent
    to the 0. There are 2, 3, or 4 possible actions depending on if 
    the empty square is in a corner, edge, or neither
    """
    N = 3
    N2 = 9
    corners = set((0, N-1, N*(N-1), N2-1))
    verbosity = 1
    digits = int((N2-1)/10) + 1

    def __init__(self, initial=None, goal=tuple(range(N2)), verbosity=None):  # , graph=nx.Graph()):
        """The constructor specifies the initial state, and possibly a goal
        state, if there is a unique goal.  Your subclass's constructor can add
        other arguments."""
        if initial:
            if isinstance(initial, (int, float)):
                self.N2 = int(initial)
                initial = None
            else:
                self.N2 = len(initial)
        else:
            self.N2 = 9
        # TODO: allow non-square boards
        self.N = int(self.N2 ** .5)
        self.digits = int((self.N2-1)/10)
        self.initial = range(self.N2)
        self.initial = initial or shuffled(range(self.N2))
        self.goal = goal or list(range(self.N2))
        self.typ = type(goal) if goal else type(initial) if initial else tuple
        if verbosity is not None:
            self.verbosity = int(verbosity)
        super(NPuzzleProblem, self).__init__(initial=self.initial, goal=self.goal, verbosity=self.verbosity)


    def actions(self, state, check_reversible=False):
        if self.verbosity:
            print('actions({0}). . .'.format(self.pretty_str(state)))
        i0 = state.index(0)
        # 4 corners
        if i0 == 0:                    # top-left
            possibilities = [swap(state, i0, i0 + 1), swap(state, i0, i0 + self.N)]
        elif i0 == self.N - 1:         # top-right
            possibilities = [swap(state, i0, i0 - 1), swap(state, i0, i0 + self.N)]
        elif i0 == self.N2 - self.N:   # bottom-left
            possibilities = [swap(state, i0, i0 + 1), swap(state, i0, i0 - self.N)]
        elif i0 == self.N2 - 1:        # bottom-right
            possibilities = [swap(state, i0, i0 - 1), swap(state, i0, i0 - self.N)]
        elif i0 < self.N:              # noncorner, top edge  (3 actions)
            possibilities = [swap(state, i0, i0 - 1), swap(state, i0, i0 + 1), swap(state, i0, i0 + self.N)]
        elif i0 > self.N2 - self.N:    # noncorner, bottom edge  (3 actions)
            possibilities = [swap(state, i0, i0 - 1), swap(state, i0, i0 + 1), swap(state, i0, i0 - self.N)]
        elif not i0 % self.N:          # noncorner left edge
            possibilities = [swap(state, i0, i0 - self.N), swap(state, i0, i0 + self.N), swap(state, i0, i0 + 1)]
        elif i0 % self.N == (self.N - 1):          # noncorner right edge
            possibilities = [swap(state, i0, i0 - self.N), swap(state, i0, i0 + self.N), swap(state, i0, i0 - 1)]
        # noncorner, nonedge
        else:
            possibilities = [swap(state, i0, i0 - self.N), swap(state, i0, i0 + self.N), swap(state, i0, i0 - 1), swap(state, i0, i0 + 1)]
        if self.verbosity:
            print('possibilities = {0}'.format(possibilities))
        # all actions should be reversible
        if check_reversible:
            for act in possibilities:
                if self.verbosity:
                    print('assert({0} in self.actions({1}) ({2}) )'.format(state, act, self.actions(act, check_reversible=False)))
                assert(state in self.actions(act, check_reversible=False))
        return possibilities

    def pretty_str(self, state):
        return str(state)
        s = ('_' * (self.digits + 1))*self.N + '\n'
        eol = '\n'
        for row in range(self.N):
            s += ' '.join(('{0: '+str(self.digits)+'d}').format(int(state[self.N * row + col])) if row+col else ' '*self.digits for col in range(self.N)) + eol
        s += ('-' * (self.digits + 1))*self.N + '\n'
        return s


def distance((ax, ay), (bx, by)):
    "The distance between two (x, y) points."
    return math.hypot((ax - bx), (ay - by))


def h_npuzzle_simple(node, N2=None):
    state = getattr(node, 'state', node)
    N2 = N2 or h_npuzzle_simple.N2 or max(state) + 1
    N = int(N2 ** 0.5)
    h_npuzzle_manhattan.N2 = N2
    #    moves so far + horizontal distance     + vertical distance
    return node.depth + node.state.index(0) % N + node.state.index(0) / 3
h_npuzzle_simple.N2 = None


def h_npuzzle_manhattan(node, N2=None, verbosity=None):
    """Distance of all numbered tiles from their goal position, summed

    Interestingly, this ignores the distance from the goal for the vacant
    tile (hole or gap).

    References:
        http://heuristicswiki.wikispaces.com/Manhattan+Distance
    """
    depth = node.depth
    state = getattr(node, 'state', node)
    N2 = N2 or h_npuzzle_manhattan.N2 or max(state) + 1
    h_npuzzle_manhattan.N2 = N2
    N = int(N2 ** 0.5)
    if verbosity is None:
        verbosity = h_npuzzle_manhattan.verbosity
    else:
        verbosity = verbosity
        h_npuzzle_manhattan.verbosity = verbosity
    distance = 0 
    for tile in range(1, N2):
        pos = state.index(tile)
        distance += abs(pos % N - tile % N) + abs(pos / N - tile / N)
    if verbosity:
        print('h_npuzzle_manhattan(node.state={0}, N2={1}) distance = {2}'.format(state, N2, distance))
    return depth + distance 
h_npuzzle_manhattan.N2 = None
h_npuzzle_manhattan.verbosity = 0


def astar_search(problem, heuristic=h_npuzzle_manhattan):
    """Modified version of Norvig's A* graph search algorithm

    Allows unhashable states by converting them to hashable types inside Node

    Search the nodes with the lowest heuristic scores first.
    You specify the function heuristic(node) that you want to minimize; for example,
    if heuristic is a heuristic estimate to the goal, then we have greedy best
    first search; if heuristic is node.depth then we have breadth-first search.
    There is a subtlety: the line "heuristic = memoize(heuristic, 'heuristic')" means that the heuristic
    values will be cached on the nodes as they are computed. So after doing
    a best first search you can examine the heuristic values of the path returned."""
    # heuristic = memoize(heuristic, 'heuristic')
    node = Node(problem.initial)
    if problem.goal_test(node):
        return node
    frontier = PriorityQueue(min, heuristic)
    frontier.append(node)
    explored = set()
    while frontier:
        node = frontier.pop()
        if problem.goal_test(node):
            return node, explored
        explored.add(force_hashable(node.state))
        for child in node.expand(problem):
            hashable_child = force_hashable(child.state)
            if hashable_child not in explored and child not in frontier:
                frontier.append(child)
            elif hashable_child in frontier:
                hashable_incumbent = frontier[hashable_child]
                if heuristic(hashable_child) < heuristic(hashable_incumbent):
                    del frontier[hashable_incumbent]
                    frontier.append(hashable_child)
    return None, explored



# class WordNetProblem(Problem):
#     "The problem of searching a graph from one node to another."
#     def __init__(self, initial='bacon', goal='dancing'):
#         Problem.__init__(self, initial, goal)
#         self.graph = wn

#     def actions(self, word):
#         """The actions at a word are just its "neighbors", synonyms, 
#         alternatively the other words in the categories it belongs."""
#         # if isinstance(word, nltk.corpus.reader.wordnet.Synset)
#         return list(ss.split('.')[0] for sss in wn.synsets(word) for ss in sss)
#         # return self.graph.get(A).keys()

#     def result(self, state, action):
#         "The result of going to a neighbor is just that neighbor."
#         return action

#     def path_cost(self, cost_so_far, A, action, B):
#         return cost_so_far + (self.graph.get(A,B) or infinity)

#     def h(self, node):
#         "h function is straight-line distance from a node's state to goal."
#         locs = getattr(self.graph, 'locations', None)
#         if locs:
#             return int(distance(locs[node.state], locs[self.goal]))
#         else:
#             return infinity



# class WordNetProblem(Problem):
#     G = Graph()
#     start = 'bacon'
#     goal = 'dance'

#     def __init__(self, start='bacon', goal='dance'):
#         self.start, self.goal = start, goal

#     def actions():



# def graph_for_problem(name=None):
#     """Formulate a problem as a networkx Graph with a goal nodes named 'goal' and start nodes named 'start'"""
#     if name in (None, '') or name.lower().strip() in ('', 'null', 'none', 'empty'):
#         name = 'No Problem'.format('N')
#         G = nx.Graph(name=str(name))
#         G.add_node(0, name='start')
#         G.add_node(1, name='goal')
#         G.add_edge(0, 1, action='move')
#         return G
#     name = str(name).lower().strip()
#     if re.match(r'[1-8]{1,2}[-\s]?puz'):
#         N = int(name[0]) + 1
#         M = sqrt(N)
#         assert(M**2 == N)
#         name = '{0} Puzzle'.format(N)
#         initial_positions = np.random.permutation(N)
#         G = nx.Graph(name=str(name))
#         G.add_node(0, name='start')
#         G.add_node(1, name='goal')
#         G.add_edge(0, 1, action='move')
#         return G
#     if 'mission' in name or 'cani' in name:
#         return G 


# def search_strategy(frontier):
#     """Return the next node in the frontier to explore"""
#     return frontier[0]


# def path_to_node(node):
#     pass


# G = graph_for_problem('null')


# def plan_path_to_goal(G):
#     """Return a list of edges in graph G that connect the node named 'start' to any node named 'goal'"""
#     frontier = search_nodes(G.nodes(name='start'))
#     node = select_next_node(frontier, strategy)
#     if not frontier:
#         return None
#     if node.name == 'goal':
#         return path_to_node(node)
#     fringe += [node]


def nodes_at_depth(problem, initial=None, depth=27, verbosity=1):
    """Breadth-first search to see how many unique states exist at a given distance/depth from `initial`"""
    if initial is None:
        initial = problem.goal
    visited = set()  # set of hashable states already visited
    frontier = set()   # set of hashable states at fringe being explored now
    node = Node(problem.initial)
    frontier.add(node)
    d = 0
    while frontier and d < depth:
        if verbosity:
            print('='*40 + ' {0:03d}'.format(d) + ' ' + '='*40)
        d += 1
        children = set()  # set of nodes to explore next
        for node in frontier:
            hashable_state = force_hashable(node.state)
            visited.add(hashable_state)
            for child in node.expand(problem):
                hashable_child = force_hashable(child.state)
                if hashable_child not in visited:
                    children.add(child)
        frontier = set(children)
        if verbosity > 1:
            print(','.join(''.join(str(i) for i in node.state) for node in frontier))
        elif verbosity:
            print(len(children))
    return frontier

initial = swap(range(9), 0, 1)
prob = NPuzzleProblem(initial=initial, verbosity=0)
goal_2step, explored_1step = astar_search(prob, heuristic=h_npuzzle_manhattan)
assert(len(goal_2step.path()) == 2)
# print(goal_node.path())


def compare():
    initial = swap(swap(swap(swap(swap(range(9), 0, 1), 1, 2), 2, 5), 5, 4), 4, 7)
    prob = NPuzzleProblem(initial=initial, verbosity=0)
    goal_manhat, explored_manhat = astar_search(prob, heuristic=h_npuzzle_manhattan)
    # print(goal_manhattan.path())

    prob = NPuzzleProblem(initial=initial, verbosity=0)
    goal_simple, explored_simple = astar_search(prob, heuristic=h_npuzzle_simple)
    # print(goal_simple.path())
    return goal_manhat, explored_manhat, goal_simple, explored_simple

