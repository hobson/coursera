from collections import OrderedDict

class NormalForm:
	"""Matrix of game outcomes, utility values for each player (game theory Normal Form)

	TODO:
		expand to more than 2 players (greater than 2-D matrix)

	>>> prisoners_dilemma = NormalForm([[(-1,-1),(-3,0)],[(0,-3),(-2,-2)]])
	>>> prisoners_dilemma.dominant(0)
	(-2, 1, 1)
	>>> prisoners_dilemma.dominant(1)
	(-2, 1, 1)
	>>> game = [[(1,2), (2,2), (5,1)],
				[(4,1), (3,5), (3,3)],
				[(5,2), (4,4), (7,0)],
				[(2,3), (0,4), (3,0)],
				]
	"""
	u = None
	u_min = None

	def __init__(self, *args):
		self.u = None
		if (args and args[0] and args[0][0] and isinstance(args[0], (list, tuple)) 
			and args[0][0] and isinstance(args[0][0], (list, tuple))
			and len(args[0][0][0]) == 2):
			self.u = []
			for row in args[0]:
				self.u += [tuple(tuple(payoffs) for payoffs in row)]
			self.u = tuple(self.u)

	def pareto_optimima(self):
		pass

	def nash_equilibirum(self):
	 	pass

	def dominant(self, player=None):
		"""Find the strategy minimax--the highest payoff if an oponent choses the strategy that minimizes your payoff for that choice"""
		worsts =[]

		if player is None:
			return max_with_index(self.dominant(player=p) for p in [0, 1]) 
		elif player is 0:
			for i, payoffs in enumerate(self.u):
				worst = min_with_index(u_r for u_r, u_c in payoffs)
				worsts += [(worst[0], i, worst[1])]
			worsts = sorted(worsts)
			return worsts[-1]
		elif player is 1:
			for j in range(len(self.u[0])):
				payoffs_col = [payoffs[j] for payoffs in self.u]
				worst = min_with_index(u_c for u_r, u_c in payoffs_col)
				worsts += [(worst[0], worst[1], j)]
			worsts = sorted(worsts)
			return worsts[-1]
		return worsts

	def __repr__(self):
		lines = '\n'.join('%s' % list(l) for l in list(self.u,))
		if len(self.u) <= 1:
			return 'NormalForm(%s)' % lines
		return 'NormalForm(\n%s)' % lines


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