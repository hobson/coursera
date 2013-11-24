# cross product of all the actions for each node
pure_strategies = []
for player, nodes in enumerate((('N', 'P'), ('St', 'Sh')), (('T', 'D'),)):
    pure_strategies += [[]]
    # FIXME need to do something else for the cross product across nodes, needs recursion or iteration over combinations
    for node, actions in enumerate(nodes):
        for action in actions:
            pure_strategies[node] += [(action, node)]
pure_strategies = [[('N', 'St'), ('N', 'Sh'), ('P', 'St'), ('P', 'Sh')],
                   [('T',), ('D',)]]

            # ('T',),   ('D',)
payoff = [
            #('N', 'St')
            [(1,  1),  (1, 1)],
            #('N', 'Sh')
            [(1,  1),  (1, 1)],
            #('P', 'St')
            [(10, 0),  (0, 2)],
            #('P', 'Sh')
            [(5,  5),  (0, 2)],
         ]
