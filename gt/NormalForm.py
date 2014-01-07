#from collections import OrderedDict


class NormalForm:
    """Matrix of game outcomes, utility values for each player (game theory Normal Form)

    TODO:
        expand to more than 2 players (greater than 2-D matrix)
        implement nash equilibria solution finder
        implement bayes equiplibria solution finder
        implement mixed strategy equilibria finders
        implement symbolic equations that find equilibria probability inflection points for bayesean games

    >>> prisoners_dilemma = NormalForm([[(-1,-1),(-3,0)],[(0,-3),(-2,-2)]])
    >>> prisoners_dilemma.dominant(0)
    (-2, 1, 1)
    >>> prisoners_dilemma.dominant(1)
    (-2, 1, 1)
    >>> game = [[(1, 2), (2, 2), (5, 1)],\
                [(4, 1), (3 ,5), (3, 3)],\
                [(5, 2), (4, 4), (7, 0)],\
                [(2, 3), (0, 4), (3, 0)],\
                ]
    >>> NormalForm(game).dominant()
    [{'player 0 (payoff, strategy)': (4, 2)}, {'player 1 (payoff, strategy)': (2, 0)}]
    """
    def __init__(self, *args):
        self.u = None
        self.players = None
        if (args and args[0] and args[0][0] and isinstance(args[0], (list, tuple))
                and args[0][0] and isinstance(args[0][0], (list, tuple))
                and len(args[0][0][0]) == 2):
            self.u = []
            for row in args[0]:
                self.u += [tuple(tuple(payoffs) for payoffs in row)]
            self.u = tuple(self.u)
        if (len(args) > 0 and len(args[1]) >= 2 and all(isinstance(a, basestring) for a in args[1])):
            self.players = [str(a).strip() for a in args[1]]
        else:
            self.players = ['player_%d' % i for i in range(1, len(self.u) + 1)]

    def pareto_optimima(self):
        """The strategy profiles for all players that maximize the total good

        One strategy for each player, one profile or strategies for each pareto optimum.
        No player can be given any more utility/payoff without reducing the payoff to other players."""
        pass

    def nash_equilibirum(self):
        """The stratgy profiles for all players that they play the best response to their oponent

        No player can achieve any more payoff for the given strategy of their oponent.
        Algoirthm:
            1. eliminate all strictly dominated strategies
            2. pick any weakly dominant strategy for player 1
            3. find the best response for player 2
            4. find the best response for player 1, changing strategies only if a strictly better payoff is found
            5. repeat 3-4 until a cycle or equilibrium is found
            6. repeat 2-5 until all weakly dominant strategies for player 1 have been considered
            7. repeat 2-5 starting with a different "player 1" until all players have had their chance at "poll position"
            8. the set of strategy profiles for each player produced in step 5 are Nash Equilibria
        """
        pass

    def dominant(self, player=None, weakly=False):
        """Find the maxmin strategy that dominates all others.

        This is the best worst option for the player selected.
        This is the strategy with the highest payoff (utility), when opponents are
          attempting to minimize oponents' payoff.
        This is equivalent to the nash equilibirum iff an opponent's payoff (utility) is inversely
          proportional to their oponents' payoff (zero-sum game).
        """
        worsts = []

        # if no player is identified, then run for both players
        if player is None:
            return [{'%s (payoff, strategy)' % p: self.dominant(player=i, weakly=weakly)} for i, p in enumerate(self.players)]
        elif player in (0, self.players[0]):
            for i, payoffs in enumerate(self.u):
                worst = min_with_index(u_r for u_r, u_c in payoffs)
                worsts += [(worst[0], i, worst[1])]
            worsts = sorted(worsts)
            return (worsts[-1][0], self.player[worsts[-1][1]])
        elif player in (1, self.players[1]):
            for j in range(len(self.u[0])):
                payoffs_col = [payoffs[j] for payoffs in self.u]
                worst = min_with_index(u_c for u_r, u_c in payoffs_col)
                worsts += [(worst[0], worst[1], j)]
            worsts = sorted(worsts)
            return (worsts[-1][0], self.player[worsts[-1][1]])

    def dominated(self, player=None, weakly=False):
        """Find all strategies which are dominated by other strategies """

    def __repr__(self):
        lines = '\n'.join('%s' % list(l) for l in list(self.u,))
        if len(self.u) <= 1:
            return '%s(%s)' % (self.__class__.__name__, lines)
        return '%s(\n%s)' % (self.__class__.__name__, lines)


def min_with_index(seq):
    mn = float('inf')
    mn_index = None
    for i, val in enumerate(seq):
        if val < mn:
            mn = val
            mn_index = i
    return mn, mn_index


def max_with_index(seq):
    mx = None
    mx_index = None
    for i, val in enumerate(seq):
        if val > mx:
            mx = val
            mx_index = i
    return mx, mx_index
