"""groves module

classes for implementing various Groves Mechanisms, including Vickrey-Clarke-Groves (VCG)
as described in Coursera Game Theory 2, week 3.
"""

import numpy as np

class VCG(object):
    """
    Computes the winner and payments for a Vickerey-Clarke-Groves game Mechanism

    >>> vcg = VCG([[0, 60], [45, 15], [45, 5]], outcome_labels=['build', 'do not build'])
    >>> vcg.payments
    [0, 20, 30]
    >>> vcg.outcomes
    [0, 1]
    >>> vcg.get_outcomes(label=True)
    ['build', 'do not build']
    >>> vcg = VCG(np.array([[3, 3, 0, 0], [0, 0, 2, 2], [2, 0, 0, 0], [0, 1, 0, 0], [0, 0, 5, 0], [0, 0, 0, 3], [2, 0, 0, 0], [0, 1, 0, 1]]) * -1, agent_labels=['AB', 'AC', 'BD', 'BE', 'CF', 'CE', 'DF', 'EF'], outcome_labels=['ABDF', 'ABEF', 'ACF', 'ACEF'])
    >>> vcg.payments
    [-4, 1, 0, 0, 3, 2, 0, 0]
    >>> vcg.outcomes
    [1, 3, 2, 0]
    >>> vcg = VCG([[10, -15, 0], [-12, 5, 0], [0, 0, 4]])
    >>> vcg.outcomes
    [2, 0, 1]
    >>> vcg.payments
    [1, 6, 0]
    """
    def __init__(self, valuations, agent_labels=None, outcome_labels=None):
        self.payments, self.outcome, self.outcomes, self.total_valuation = None, None, None, None
        self.valuations = np.array(valuations)
        self.total_valuation = None
        self.agent_labels = agent_labels or [str(i + 1) for i in range(len(valuations))]
        self.outcome_labels = outcome_labels or [str(i + 1) for i in range(len(valuations))]
        self.N = len(valuations)  # number of players
        self.M = max(len(utilities) for utilities in valuations)  # number of possible outcome_labels (choices)
        self.compute_outcomes_and_payments()
    
    def get_outcomes(self, only_one=False, label=False):
        if not self.outcomes:
            self.compute_outcomes_and_payments()
        if only_one:
            oc = self.outcomes[0]
            if label:
                return self.outcome_labels[oc]
            return oc
        return self.outcomes

    def agent_index(self, agent):
        if agent in self.agent_labels:
            return self.agent_labels.index(agent)
        elif agent in range(self.N):
            return agent
        raise ValueError("The argument to agent_index(), `agent`, should be the 0-offset index of an agent or an agent's string label.")

    def compute_outcomes_and_payments(self, agent=None):
        if self.payments is not None:
            return self.payments
        self.total_valuation = self.valuations.sum(axis=0)
        self.outcomes = [j for val, j in sorted([(value, i) for (i, value) in enumerate(self.total_valuation)], reverse=True)]
        self.outcome = self.outcomes[0]
        self.payments = [-sum(self.valuations[j][self.outcome] for j in range(self.N) if j != excluded) for excluded in range(self.N)]
        print self.payments
        # TODO: do this with numpy slices or np.array.dot products with sequences of 0's and 1's
        for i in range(self.N):
            print i, self.agent_labels[i], self.outcome_labels[self.outcome_without_one(i)]
            self.payments[i] += sum(self.valuations[j][self.outcome_without_one(i)] for j in range(self.N) if j !=i)
        self.payments = self.payments or self.get_payments()
        return self.outcomes, self.payments

    def outcome_without_one(self, agent):
        agent = self.agent_index(agent)
        if self.total_valuation is None:
            self.total_valuation = self.valuations.sum(axis=0)
        return max((value, j) for (j, value) in enumerate(self.total_valuation - np.array(self.valuations[agent])))[1]

    def get_payment(self, agent):
        """Get the payment that a particular agent must make to the mechanism "bank").

        A negative payment indicates a payment received by the agent.
        """
        agent = self.agent_index(agent)
        if self.payments is None:
            self.payments = self.get_payments()
        return self.payments[agent]


class GraphVCG(VCG):
    """Compute valuations (utilities) based on edge costs (rather than agent utilities).

    agent_labels are the unique, ordered pair of symbols that define a graph edge, 
        e.g. 'AB' for the edge from node A to node B
    outcome_labels are the ordered strings of node symbols that represent a valid outcome,
        e.g. 'ABC', where C is a final node, or accepting state node

    >>> vcg = GraphVCG(agent_costs=[3, 2, 2, 1, 5, 3, 2, 1], agent_labels=['AB', 'AC', 'BD', 'BE', 'CF', 'CE', 'DF', 'EF'], outcome_labels=['ABDF', 'ABEF', 'ACF', 'ACEF'])
    >>> vcg.payments
    [0, 1, 0, 0, 3, 2, 0, 0]
    >>> vcg.outcomes
    [1, 3, 2, 0]
    """
    def __init__(self, agent_costs=None, agent_labels=None, outcome_labels=None):
        raise NotImplementedError('outcome_without_one needs to exclude solutions that pass through the agent being excluded.')
        valuations = [[-1 * int(agent_labels[i] in outcome_labels[j]) * agent_costs[i] for j in range(len(outcome_labels))] for i in range(len(agent_labels))]
        super(GraphVCG, self).__init__(valuations, agent_labels, outcome_labels)

    def outcome_without_one(self, agent):
        raise NotImplementedError('outcome_without_one needs to exclude solutions that pass through the agent being excluded.')
        agent = self.agent_index(agent)
        if self.total_valuation is None:
            self.total_valuation = self.valuations.sum(axis=0)
        for value in self.total_valuation - np.array(self.valuations[agent]):
            pass
