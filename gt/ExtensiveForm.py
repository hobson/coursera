# pirates 
# captain choses divition of 100 coins (A is first captain)
# if captain wins votes of a majority of pirates, including himself then everyone gets the proposed division, otherwise captain dies
# when capatin dies next alphabetical pirate is appointed captain and game repeats

players = 'ABCDE'
actions = list(range(101))

# backward induction:
# D vs E: D can propose 100 for himself and 0 for E and will get it because 1/2 votes is sufficient
# C vs DE: D's options are 100 or whatever C proposes, so he'll vote to throw C overboard regardless (indifference to 100 coin offer either way), E's options are 0 or whatever captain proposes so he'll take anything, but vote for mutiny if offered 0, so C knows that E will vote for mutiny (indifference to 0 payoff) so he can propose 1 to E and 99 for self and 0 to D and win it
# B vs CDE: D will accept 1, C accept 100, E accept 2. So B will offer 1 to D and take 99 so he'll have the majority
# A vs BCDE: D will accept 2, C accept 1, E accept 1, B accept 100, so A offers 1 to C and E and keeps 98


# each node is actually a Game/Subgame
class Game(object):
    
    def __init__(self, children=None, parent=None, player=0):
        self.parent = parent or None
        self.player = player or 0
        # can be a Game or a real value, if real the child is a payoff for that player
        self.children = children or []


    def best_payoff(self):
        # FIXME: the list of children (actions) doesn't have to be the same as the number of players, and this player is only interested in *his* best payoff
        if self.children and isinstance(self.children[0], Game):
            return max(g.best_payoff() for g in self.children)

    def normalize(self):
        pass






