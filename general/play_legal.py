

class PlayLegal(object):
    
    def __init__(self, game=None, role=None, roles=None, state=None):
        self.game = game
        self.role = role
        self.roles = roles
        self.state = state

    def info():
        '''A simple legal player doesn't need to do anything to plan for a game'''
        return 'ready'
      
    def start(self, id, player, rules, sc, pc):
        '''The start event handler assigns values to game and role based on the incoming start message; it uses findroles to compute the roles in the game; it uses findinits to compute the initial state; and it returns ready, as required by the GGP protocol.
        '''
        self.game = rules
        self.role = player
        self.roles = findroles(self.game)
        self.state = findinits(self.game)
        return 'ready'
        
    def play(self, id, move):
        '''The play event handler takes a match identifier and a move as arguments. It first uses the simulate subroutine to compute the current state. If the move is nil, the subroutine returns the current state. Otherwise, it uses findnext to compute the state resulting from the specified move and the specified state. Once our player has the new state, it uses findlegalx to compute a legal move.
        '''
        self.state = simulate(move, self.state)
        return findlegalx(self.role, self.state, self.game)

    def simulate(self, move, state):
        if move=='nil':
            return state
        return findnext(self.roles, move, state, self.game)
    
    def abort(self, id):
        '''The abort event handler for our player does nothing. It ignores the inputs and simply returns done as required by the GGP protocol.'''
        return 'done'
        
    def stop(self, id, move):
        '''Like the abort message handler, the stop event handler for our legal player also does nothing. It simply returns done.'''
        return 'done'


def findlegalx(role, state, game):
    pass

def findnext(roles, move, state, game):
    pass

def simulate(move, state):
    pass

def findroles(game):
    pass

def findinits(game):
    pass