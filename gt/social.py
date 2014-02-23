"""Game Theory social welfare and social choice functions (SCWs and SCFs)

Implements the following mechanisms:
* Plurality (conventional voting)
* Borda-Rule
* Pairwise Elimination
* Plurality with Elimination (similar to French presidential election)
* Condorcet choice (not yet validated or tested)

Agent preferences are specified as a list of lists.
Agents are just given index numbers (row index in the list),
But candidates are given whatever labels are listed in each row of the preferences argument.

Arbitrary agent "influence" weights are allowed (for semidictatorship).
Arbitrary candidate weights are allowed (advantage if weight > 1 or disadvantage if weight < 1)
Arbitrary rank weights are allowed (for nonlinear Borda-Rule scoring "kernels")

FIXME:
- Many functions and have a tie_breaker argument that doesn't do anything. Use it or lose it!
- Various TODOs sprinkled in code comments

"""

import warnings
A, B, C, D = 'A', 'B', 'C', 'D'



def tie_breaker_choice(names, method='name', preferences=None, votes=None):
    if method.lower().strip() == 'name':
        return sorted(names)[0]


def plurality_choice(preferences, agent_weights=None, candidate_weights=None, tie_breaker='name'):
    """"Return the winner using the plurality voting method (most prefered by most agents)

    >>> plurality_choice(((A, B, D, C), (D, C, B, A), (B, D, C, A), (C, A, B, D), (C, D, A, B)))
    ('C', 2)
    >>> plurality_choice([(A, B, D, C)] * 400 + [(D, C, B, A)] * 300 + [(B, D, C, A)] * 200 + [(C, A, B, D)] * 100 + [(C, D, A, B)] * 2)
    ('A', 400)
    >>> plurality_choice(((A, B, D, C), (D, C, B, A), (B, D, C, A), (C, A, B, D), (C, D, A, B)), agent_weights=(400, 300, 200, 100, 2))
    ('A', 400)
    >>> plurality_choice([(A, B, D, C)] * 300 + [(D, C, B, A)] * 400 + [(B, D, C, A)] * 200 + [(C, A, B, D)] * 100 + [(C, D, A, B)] * 2)
    ('D', 400)
    """
    #print candidate_weights
    ranking, score = plurality_welfare(preferences, agent_weights=agent_weights, candidate_weights=candidate_weights, tie_breaker=tie_breaker)
    #print ranking, score
    # winners = [ranking[0]]
    # for i, c in enumerate(ranking[1:]):
    #     if c not in score or winners[0] not in score or score[c] > score[winners[0]]:
    #         raise RuntimeError('The plurality_welfare() function output an invalid ranking list or score dict. Candidate %s had a score of %s and rank %s which exceeds the 1st ranked candidate, %s, with a score of %s'
    #             (c, score[c], ranking.index(c) + 1, winners[0], score[winners[0]]))
    #     elif score[c] == score[winners[0]]:
    #         winners += [c]
    #     else:
    #         break
    return ranking[0], score[ranking[0]]


def plurality_welfare(preferences, agent_weights=None, candidate_weights=None, tie_breaker='name'):
    """"Return the winner using the plurality voting method (most prefered by most agents)

    >>> plurality_welfare(((A, B, D, C), (D, C, B, A), (B, D, C, A), (C, A, B, D), (C, D, A, B)))
    (('C', 'D', 'B', 'A'), {'A': 1, 'C': 2, 'B': 1, 'D': 1})
    >>> plurality_welfare([(A, B, D, C)] * 400 + [(D, C, B, A)] * 300 + [(B, D, C, A)] * 200 + [(C, A, B, D)] * 100 + [(C, D, A, B)] * 2)[0]
    ('A', 'D', 'B', 'C')
    >>> plurality_welfare(((A, B, D, C), (D, C, B, A), (B, D, C, A), (C, A, B, D), (C, D, A, B)), agent_weights=(400, 300, 200, 100, 2))
    (('A', 'D', 'B', 'C'), {'A': 400, 'C': 102, 'B': 200, 'D': 300})
    >>> plurality_welfare([(A, B, D, C)] * 300 + [(D, C, B, A)] * 400 + [(B, D, C, A)] * 200 + [(C, A, B, D)] * 100 + [(C, D, A, B)] * 2)
    (('D', 'A', 'B', 'C'), {'A': 300, 'C': 102, 'B': 200, 'D': 400})
    """
    N = get_ranking_len_from_preferences(preferences)
    rank_weights = [1] + [0] * (N - 1)
    return borda(preferences, agent_weights=agent_weights, candidate_weights=candidate_weights, rank_weights=rank_weights)


def plurality_choice_with_elimination(preferences, agent_weights=None, candidate_weights=None, tie_breaker='name'):
    """Return the winning candidate name and its score in a plurality with elimination social choice function 
    
    >>> plurality_choice_with_elimination(((A, B, D, C), (D, C, B, A), (B, D, C, A), (C, A, B, D), (C, D, A, B)))
    ('C', 3)
    >>> plurality_choice_with_elimination([(A, B, D, C)] * 400 + [(D, C, B, A)] * 300 + [(B, D, C, A)] * 200 + [(C, A, B, D)] * 100 + [(C, D, A, B)] * 2)
    ('D', 502)
    >>> plurality_choice_with_elimination(((A, B, D, C), (D, C, B, A), (B, D, C, A), (C, A, B, D), (C, D, A, B)), agent_weights=(400, 300, 200, 100, 2))
    ('D', 502)
    """
    while 1:
        ranking, scores = plurality_welfare(preferences, agent_weights=agent_weights, candidate_weights=candidate_weights, tie_breaker=tie_breaker)
        if len(ranking) <= 2:
            break
        eliminated_candidate = ranking[-1]
        reduced_preferences = []
        for plist in preferences:
            reduced_preferences += [tuple(pref for pref in plist if pref != eliminated_candidate)]
        preferences = reduced_preferences
    return ranking[0], scores[ranking[0]]


def borda(preferences, agent_weights=None, rank_weights=None, candidate_weights=None, tie_breaker='name'):
    """Calculate the rank, and scores of a Borda selection from ordered lists of preferences (first preferred over last).

    >>> rank, score = borda(((B, C, A, D), (B, D, C, A), (D, C, A, B), (A, D, B, C), (A, D, C, B)))
    >>> print rank
    ('D', 'A', 'B', 'C')
    >>> print [score[candidate] for candidate in rank]
    [9, 8, 7, 6]
    >>> prefs = [(A, B, D, C)] * 400 + [(D, C, B, A)] * 300 + [(B, D, C, A)] * 200 + [(C, A, B, D)] * 100 + [(C, D, A, B)] * 2
    >>> ranking, scores = borda(prefs)
    >>> ranking
    ('B', 'D', 'A', 'C')
    >>> scores == {'A': 1402, 'B': 1800, 'C': 1106, 'D': 1704}
    True
    """
    # if candidate_weights:
    #     raise NotImplementedError('borda rule SWF is not implemented for nonunity candidate_weights.')
    scores = {}
    N = get_ranking_len_from_preferences(preferences)
    candidate_weights = candidate_weights or dict((candidate, 1) for candidate in get_candidates_from_preferences(preferences))
    rank_weights = rank_weights or list(range(N - 1, -1, -1))
    agent_weights = agent_weights or [1] * len(preferences)
    for agent, plist in enumerate(preferences):
        for rank, candidate in enumerate(plist):
            scores[candidate] = scores.get(candidate, 0) + candidate_weights[candidate] * agent_weights[agent] * rank_weights[rank]
    return tuple(c2 for(s2, c2) in sorted([(s1, c1) for c1, s1 in scores.items()], reverse=True)), scores


def pairwise_elimination_choice(preferences, agenda=None, agent_weights=None, candidate_weights=None, tie_breaker='name'):
    """Return the winning candidate name and its score in a plurality with elimination social choice function 
    
    # TODO: feature to specify agenda as a string indicating how to compute an agenda:
            'name' or 'reverse name' of 'alpha' or 'reverse alpha' 
            or an agent name (to optimize the agenda for their preferences)
            'occurence in preferences' (the current default) or 'reverse occurence in preferences'

    >>> agent_weights = (400, 300, 200, 100, 2)
    >>> prefs = ((A, B, D, C), (D, C, B, A), (B, D, C, A), (C, A, B, D), (C, D, A, B))
    >>> pairwise_elimination_choice(prefs, agent_weights=agent_weights)
    <string>:152: UserWarning: You probably want to specify an agenda. Currently using a default agenda derived from candidate occurences in the preferences lists supplied: ['A', 'B', 'D', 'C']
    ('D', 900)
    >>> pairwise_elimination_choice(((A, B, D, C), (D, C, B, A), (B, D, C, A), (C, A, B, D), (C, D, A, B)), agenda=[A, B, C, D])
    ('D', 3)
    >>> pairwise_elimination_choice(((A, B, D, C), (D, C, B, A), (B, D, C, A), (C, A, B, D), (C, D, A, B)), agenda=[A, B, D, C])
    ('D', 3)
    >>> pairwise_elimination_choice([(A, B, D, C)] * 400 + [(D, C, B, A)] * 300 + [(B, D, C, A)] * 200 + [(C, A, B, D)] * 100 + [(C, D, A, B)] * 2, agenda=[D, B, A, C])
    ('C', 602)
    >>> pairwise_elimination_choice(((A, B, D, C), (D, C, B, A), (B, D, C, A), (C, A, B, D), (C, D, A, B)), agent_weights=(400, 300, 200, 100, 2), agenda=[D, B, A, C])
    ('C', 602)
    """
    candidates = get_candidates_from_preferences(preferences)
    if agenda:
        _agenda = list(agenda)  # create a new list in case the user assigned it in the function call, it's being modified below!
    else:
        _agenda = None
    if _agenda is None:
        _agenda = candidates
        warnings.warn("You probably want to specify an agenda. Currently using a default agenda derived from candidate occurences in the preferences lists supplied: %s" % repr(_agenda))
    if any(agenda_item not in candidates for agenda_item in _agenda):
        raise ValueError("Your agenda, %r, contains candidate(s) (names, labels or objects) that are not in the preferences list provided.")
    if len(_agenda) < len(candidates):
        warnings.warn("Looks like you are missing some candidates from your agenda:\n%r" % _agenda)
        for candidate in candidates:
            if candidate not in _agenda:
                _agenda += [candidate]
        warnings.warn("After appending the missing candidates, the agenda being used is:\n%r" % _agenda)

    winner = _agenda[0]
    N = len(candidates)
    for round_num in range(N - 1):
        candidate_pair = winner, _agenda[round_num + 1]
        #print candidate_pair
        #candidate_weights = dict((candidate, 1) if candidate in candidate_pair else (candidate, 0) for candidate in candidates)
        if candidate_weights:
            pair_weights = dict((candidate, weight) for (candidate, weight) in candidate_weights.iteritems() if candidate in candidate_pair)
        pair_preferences, pair_weights = [], []
        for pref in preferences:
            pair_preferences += [[candidate for candidate in pref if candidate in candidate_pair]]
        #print pair_preferences
        ranking, scores = borda(pair_preferences, agent_weights=agent_weights, candidate_weights=pair_weights, tie_breaker=tie_breaker) 
        #print ranking, scores
        winner, score = ranking[0], scores[ranking[0]]
        #print winner, score
    return winner, score

# def pairwise_elimination(preferences=None, candidates_sorted=None):
#     """Return the winnin in pairwise elimination with the agenda indicated by `candidates_sorted`

#     >>> A, B, C, D = 'A', 'B', 'C', 'D'
#     >>> pairwise_elimination(((B, C, A, D), (B, D, C, A), (D, C, A, B), (A, D, B, C), (A, D, C, B)))
#     'C'
#     >>> pairwise_elimination(((B, C, A, D), (B, D, C, A), (D, C, A, B), (A, D, B, C), (A, D, C, B)))
#     'A'
#     >>> pairwise_elimination([(A, B, D, C)] * 400 + [(D, C, B, A)] * 300 + [(B, D, C, A)] * 200 + [(C, A, B, D)] * 100 + [(C, D, A, B)] * 2)
#     'D'

#     # TODO: call borda() or plurality_choice() within these loops to allow weights
#     """
#     reverse = None
#     if candidates_sorted in (None, True, False, 1, -1):
#         if candidates_sorted in (True, False, 1, -1):
#             reverse = candidates_sorted
#         candidates = get_candidates_from_preferences(preferences)
#     else:
#         candidates = candidates_sorted

#     if reverse is not None:
#         if reverse in (-1, False):
#             reverse = True
#         else:
#             reverse = False
#         candidates = sorted(candidates, reverse=reverse)

#     winner = None

#     for i, first_candidate in enumerate(candidates[:-1]):
#         votes = 0
#         for plist in preferences:
#             for preferred_candidate in plist:
#                 if preferred_candidate in (first_candidate, candidates[i + 1]):
#                     if preferred_candidate == first_candidate:
#                         votes += 1
#                     else:
#                         votes -= 1
#                     break
#         if votes > 0:
#             winner = first_candidate
#         elif votes < 0:
#             winner = candidates[i + 1]
#         else:
#             winner = first_candidate, candidates[i - 1]

#     return winner


def get_candidates_from_preferences(preferences):
    candidates = []
    for agent, plist in enumerate(preferences[:-1]):
        for order, candidate in enumerate(plist):
            if candidate not in candidates:
                candidates += [candidate]
    return candidates


def get_ranking_len_from_preferences(preferences):
    try:
        return max(len(p) for p in preferences)
    except:
        return max(len(list(p)) for p in preferences)


def condorcet_winner(preferences=None):
    """Find the winner in a set of outcome preference "votes".

    >>> A, B, C, D = 'A', 'B', 'C', 'D'
    >>> condorcet_winner(((B, C, A, D), (B, D, C, A), (D, C, A, B), (A, D, B, C), (A, D, C, B)))
    >>> condorcet_winner(((A, B, C), (B, C, A), (C, B, A)))
    'B'
    """
    candidates = get_candidates_from_preferences(preferences)
    N = len(candidates)
    winners = [[None for j in range(N)] for i in range(N)]
    row_candidates = candidates[:-1]
    for i, c1 in enumerate(row_candidates):
        # a candidate vs herself isn't a real match, so counted as a win
        winners[i][i] = c1
        # for matches against all other candidates
        for j, c2 in enumerate(candidates[i+1:]):
            votes = 0
            for plist in preferences:
                if plist.index(c1) < plist.index(c2):
                    votes += 1
                elif plist.index(c2) < plist.index(c1):
                    votes -= 1
            if votes > 0:  # tie goes to first
                winners[i][j + i + 1] = c1
            elif votes < 0:
                winners[i][j + i + 1] = c2
            else:
                winners[i][j + i + 1] = (c1, c2)
            # mirror the off-diagnoal matrix
            winners[j + i + 1][i] = winners[i][j + i + 1]
    condorcet_statuses = [all(cw == c1 for cw in winners[i]) for i, c1 in enumerate(candidates)]
    condorcet_winners = tuple(candidates[i] for i, status in enumerate(condorcet_statuses) if status)
    if not condorcet_winners:
        return None
    if len(condorcet_winners) == 1:
        return condorcet_winners[0]
    # there was a tie
    return condorcet_winners

# That's a good question Gabrieli, but my understanding was that video 3 was about the paradoxical outcomes of social choice function (voting schemes) that return just a winner, not social welfare functions. The questions are about winners/choices, not rankings. But I re-listened Matt's description of pairwise elimination in lecture 2, and I can see why the accepted answer is that no pairwise elimination scheme can result in the Condorcet winner losing. Matt assumes that all alternatives are considered (in his legislature example), and ignores the problem of a tie. It is definitely silly of me to assume that a tie might be resolved with a coin flip or voter seniority. But it is nonetheless a theoretically a valid pairwise elimination voting mechanism that would fail to elect the Condorcet winner: 

# So the minimum insidious example I could come up with requires 4 voters and 4 candidates:

# preferences: (A, C, B), (B, C, A), (A, C, B), (B, C, A),
# agenda: (A, B, C)

# So we vote on A vs B and it's a tie (1 vote for each). The tie can be resolved by any number of mechanisms like voter seniority/rank/skill, candidate alphabetical order, or coin toss

# In all these "flavors" of pairwise elimination, C loses and A or B wins. And I think C is the Condorcet winner, right? So this is the paradox that subsequent lectures aim to partially resolve.
