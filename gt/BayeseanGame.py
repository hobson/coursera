from collections import namedtuple
from NormalForm import NormalForm

ProbabilityGame = namedtuple('ProbabilityGame', 'p game')


class BayeseanGame(object):
    """Array of NormalForm game objects along with their probabilities

    """

    def __init__(self, *args):
        self.games = []
        for arg in args:
            if len(arg) == 2:
                try:
                    self.games += [ProbabilityGame(*[float(arg[0]), NormalForm(arg[1])])]
                except:
                    self.games += [ProbabilityGame(*[float(arg[1]), NormalForm(arg[0])])]

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

    def __repr__(self):
        return '[\n' + ',\n'.join('%s,\n%s' (g.game, g.p) for g in self.games) + '\n]'
