from __future__ import print_function
import six
import future
from queue import PriorityQueue
import math
import random
from traceback import print_exc

import networkx as nx
import re

from nltk.corpus import wordnet as wn

# NORVIG AIMA code
infinity = float('inf')

def memoize(fn, slot=None):
    """Memoize fn: make it remember the computed value for any argument list.
    If slot is specified, store result in that slot of first argument.
    If slot is false, store results in a dictionary."""
    if slot:
        def memoized_fn(obj, *args):
            if hasattr(obj, slot):
                return getattr(obj, slot)
            else:
                val = fn(obj, *args)
                setattr(obj, slot, val)
                return val
    else:
        def memoized_fn(*args):
            if not memoized_fn.cache.has_key(args):
                memoized_fn.cache[args] = fn(*args)
            return memoized_fn.cache[args]
        memoized_fn.cache = {}
    return memoized_fn


class Problem(object):
    """The abstract class for a graph search problem.
    Should override the methods `.actions()`, and `.result()`
    If result(action) == action then no need to override (this is default `result()`).
    May override `.__init__()`, `.goal_test()`, and `.path_cost()`.
    """
    def __init__(self, initial, goal=None):  # , graph=nx.Graph()):
        """The constructor specifies the initial state, and possibly a goal
        state, if there is a unique goal.  Your subclass's constructor can add
        other arguments."""
        self.initial, self.goal = initial, goal

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
        return state == self.goal

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


class EightPuzzleProblem(Problem):
    """States are a sequence of the digits 0-9

    0 represents a blank. 
    So the actions available are to move a nonzero tile adjacent
    to the 0. There are 2, 3, or 4 possible actions depending on if 
    the empty square is in a corner, edge, or neither
    """
    N = 3
    N2 = 9
    corners = set((0, N-1, N*(N-1), N2-1))

    def __init__(self, initial=None, goal=tuple(range(N2))):  # , graph=nx.Graph()):
        """The constructor specifies the initial state, and possibly a goal
        state, if there is a unique goal.  Your subclass's constructor can add
        other arguments."""
        if initial:
            self.N = len(initial)
        else:
            self.N = 3 
        self.N2 = self.N * self.N
        self.initial = initial or random.permutation(list(range(self.N2)))
        self.goal = goal or list(range(self.N2))

    def actions(self, state):
        i0 = state.index(0)
        # 4 corners
        if i0 == 0:                    # top-left
            return [swap(state, i0, i0 + 1), swap(state, i0, i0 + self.N)]
        elif i0 == self.N - 1:         # top-right
            return [swap(state, i0, i0 - 1), swap(state, i0, i0 + self.N)]
        elif i0 == self.N2 - self.N:   # bottom-left
            return [swap(state, i0, i0 + 1), swap(state, i0, i0 - self.N)]
        elif i0 == self.N2 - 1:        # bottom-right
            return [swap(state, i0, i0 - 1), swap(state, i0, i0 - self.N)]
        # noncorner, top edge  (3 actions)
        elif i0 < self.N:
            return [swap(state, i0, i0 - 1), swap(state, i0, i0 + 1), swap(state, i0, i0 + self.N)]
        elif i0 > self.N2 - self.N:
            return [swap(state, i0, i0 - 1), swap(state, i0, i0 + 1), swap(state, i0, i0 - self.N)]
        elif not i0 % self.N:          # noncorner left edge
            return [swap(state, i0, i0 - self.N), swap(state, i0, i0 + self.N), swap(state, i0, i0 + 1)]
        elif not i0 % self.N:          # noncorner right edge
            return [swap(state, i0, i0 - self.N), swap(state, i0, i0 + self.N), swap(state, i0, i0 - 1)]
        # noncorner, nonedge
        return [swap(state, i0, i0 - self.N), swap(state, i0, i0 + self.N), swap(state, i0, i0 - 1), swap(state, i0, i0 + 1)]


def distance((ax, ay), (bx, by)):
    "The distance between two (x, y) points."
    return math.hypot((ax - bx), (ay - by))


def IterablePriorityQueue(PriorityQueue):
    def put():
        self.keys.append()

def astar(problem, f):
    """Search the nodes with the lowest f scores first.

    Same as Norvig implementation but uses builtin PriorityQueue and assumes
    Node(state) == state == action  (a state is it's own pointer reference)
    You specify the function f(node) that you want to minimize; for example,
    if f is a heuristic estimate to the goal, then we have greedy best
    first search; if f is node.depth then we have breadth-first search.
    There is a subtlety: the line "f = memoize(f, 'f')" means that the f
    values will be cached on the nodes as they are computed. So after doing
    a best first search you can examine the f values of the path returned."""
    raise NotImplementedError("PriorityQueue needs a __getitem__, __setitem__, and del methods before this has a chance.")
    f = memoize(f, 'f')
    node = problem.initial
    if problem.goal_test(node):
        return node
    frontier = PriorityQueue()
    frontier.put((f(node), node))
    explored = set()
    while frontier:
        node = frontier.get()
        if problem.goal_test(node):
            return node
        explored.add(node)
        for child in problem.actions(node):
            if child not in explored and child not in frontier:
                frontier.put((f(child), child))
            elif child in frontier:
                incumbent = child
                if f(child) < f(incumbent):
                    del frontier[incumbent]
                    frontier.append(child)
    return None



class GraphProblem(Problem):
    "The problem of searching a graph from one node to another."
    def __init__(self, initial, goal, graph):
        Problem.__init__(self, initial, goal)
        self.graph = graph

    def actions(self, A):
        "The actions at a graph node are just its neighbors."
        return self.graph.get(A).keys()

    def result(self, state, action):
        "The result of going to a neighbor is just that neighbor."
        return action

    def path_cost(self, cost_so_far, A, action, B):
        return cost_so_far + (self.graph.get(A,B) or infinity)

    def h(self, node):
        "h function is straight-line distance from a node's state to goal."
        locs = getattr(self.graph, 'locations', None)
        if locs:
            return int(distance(locs[node.state], locs[self.goal]))
        else:
            return infinity

class Node:
    """A node in a search tree. Contains a pointer to the parent (the node
    that this is a successor of) and to the actual state for this node. Note
    that if a state is arrived at by two paths, then there are two nodes with
    the same state.  Also includes the action that got us to this state, and
    the total path_cost (also known as g) to reach the node.  Other functions
    may add an f and h value; see best_first_graph_search and astar_search for
    an explanation of how the f and h values are handled. You will not need to
    subclass this class."""

    def __init__(self, state, parent=None, action=None, path_cost=0):
        "Create a search tree Node, derived from a parent by an action."
        update(self, state=state, parent=parent, action=action,
               path_cost=path_cost, depth=0)
        if parent:
            self.depth = parent.depth + 1

    def __repr__(self):
        return "<Node %s>" % (self.state,)

    def expand(self, problem):
        "List the nodes reachable in one step from this node."
        return [self.child_node(problem, action)
                for action in problem.actions(self.state)]

    def child_node(self, problem, action):
        "Fig. 3.10"
        next = problem.result(self.state, action)
        return Node(next, self, action,
                    problem.path_cost(self.path_cost, self.state, action, next))

    def solution(self):
        "Return the sequence of actions to go from the root to this node."
        return [node.action for node in self.path()[1:]]

    def path(self):
        "Return a list of nodes forming the path from the root to this node."
        node, path_back = self, []
        while node:
            path_back.append(node)
            node = node.parent
        return list(reversed(path_back))

    # We want for a queue of nodes in breadth_first_search or
    # astar_search to have no duplicated states, so we treat nodes
    # with the same state as equal. [Problem: this may not be what you
    # want in other contexts.]

    def __eq__(self, other):
        return isinstance(other, Node) and self.state == other.state

    def __hash__(self):
        return hash(self.state)



def astar_search(problem, f):
    """Search the nodes with the lowest f scores first.
    You specify the function f(node) that you want to minimize; for example,
    if f is a heuristic estimate to the goal, then we have greedy best
    first search; if f is node.depth then we have breadth-first search.
    There is a subtlety: the line "f = memoize(f, 'f')" means that the f
    values will be cached on the nodes as they are computed. So after doing
    a best first search you can examine the f values of the path returned."""
    f = memoize(f, 'f')
    node = Node(problem.initial)
    if problem.goal_test(node.state):
        return node
    frontier = PriorityQueue()
    frontier.put(node)
    explored = set()
    while frontier:
        node = frontier.pop()
        if problem.goal_test(node.state):
            return node
        explored.add(node.state)
        for child in node.expand(problem):
            if child.state not in explored and child not in frontier:
                frontier.append(child)
            elif child in frontier:
                incumbent = frontier[child]
                if f(child) < f(incumbent):
                    del frontier[incumbent]
                    frontier.append(child)
    return None

# END NORVIG AIMA code

class WordNetProblem(Problem):
    "The problem of searching a graph from one node to another."
    def __init__(self, initial='bacon', goal='dancing'):
        Problem.__init__(self, initial, goal)
        self.graph = wn

    def actions(self, word):
        """The actions at a word are just its "neighbors", synonyms, 
        alternatively the other words in the categories it belongs."""
        # if isinstance(word, nltk.corpus.reader.wordnet.Synset)
        return list(ss.split('.')[0] for ss in sss for sss in wn.synsets(word))
        # return self.graph.get(A).keys()

    def result(self, state, action):
        "The result of going to a neighbor is just that neighbor."
        return action

    def path_cost(self, cost_so_far, A, action, B):
        return cost_so_far + (self.graph.get(A,B) or infinity)

    def h(self, node):
        "h function is straight-line distance from a node's state to goal."
        locs = getattr(self.graph, 'locations', None)
        if locs:
            return int(distance(locs[node.state], locs[self.goal]))
        else:
            return infinity



# class WordNetProblem(Problem):
#     G = Graph()
#     start = 'bacon'
#     goal = 'dance'

#     def __init__(self, start='bacon', goal='dance'):
#         self.start, self.goal = start, goal

#     def actions():



def graph_for_problem(name=None):
    """Formulate a problem as a networkx Graph with a goal nodes named 'goal' and start nodes named 'start'"""
    if name in (None, '') or name.lower().strip() in ('', 'null', 'none', 'empty'):
        name = 'No Problem'.format('N')
        G = nx.Graph(name=str(name))
        G.add_node(0, name='start')
        G.add_node(1, name='goal')
        G.add_edge(0, 1, action='move')
        return G
    name = str(name).lower().strip()
    if re.match(r'[1-8]{1,2}[-\s]?puz'):
        N = int(name[0]) + 1
        M = sqrt(N)
        assert(M**2 == N)
        name = '{0} Puzzle'.format(N)
        initial_positions = np.random.permutation(N)
        G = nx.Graph(name=str(name))
        G.add_node(0, name='start')
        G.add_node(1, name='goal')
        G.add_edge(0, 1, action='move')
        return G
    if 'mission' in name or 'cani' in name:
        return G 


def search_strategy(frontier):
    """Return the next node in the frontier to explore"""
    return frontier[0]


def path_to_node(node):
    pass


G = graph_for_problem('null')


def plan_path_to_goal(G):
    """Return a list of edges in graph G that connect the node named 'start' to any node named 'goal'"""
    frontier = search_nodes(G.nodes(name='start'))
    node = select_next_node(frontier, strategy)
    if not frontier:
        return None
    if node.name == 'goal':
        return path_to_node(node)
    fringe += [node]

