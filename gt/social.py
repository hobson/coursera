
A, B, C, D = 'A', 'B', 'C', 'D'


def borda(preferences=None):
    """Calculate the rank, and scores of a Borda selection from ordered lists of preferences (first preferred over last).

    >>> A, B, C, D = 'A', 'B', 'C', 'D'
    >>> borda(((B, C, A, D), (B, D, C, A), (D, C, A, B), (A, D, B, C), (A, D, C, B)))
    None
    """
    scores = {}
    try:
        N = max(len(p) for p in preferences)
    except:
        N = max(len(list(p)) for p in preferences)
    for voter, plist in enumerate(preferences):
        for order, candidate in enumerate(plist):
            scores[candidate] = scores.get(candidate, 0) + N - order
    return tuple(c2 for(s2, c2) in sorted([(s1, c1) for c1, s1 in scores.items()], reverse=True)), scores


def pairwise_elimination(preferences=None, candidates_sorted=None):
    """Return the winnin in pairwise elimination with the agenda indicated by `candidates_sorted`

    >>> A, B, C, D = 'A', 'B', 'C', 'D'
    >>> pairwise_elimination(((B, C, A, D), (B, D, C, A), (D, C, A, B), (A, D, B, C), (A, D, C, B)))
    'C'
    >>> pairwise_elimination(((B, C, A, D), (B, D, C, A), (D, C, A, B), (A, D, B, C), (A, D, C, B)))
    'A'
    """
    reverse = None
    if candidates_sorted in (None, True, False, 1, -1):
        if candidates_sorted in (True, False, 1, -1):
            reverse = candidates_sorted
        candidates = get_candidates_from_preferences
    else:
        candidates = candidates_sorted

    if reverse is not None:
        if reverse in (-1, False):
            reverse = True
        else:
            reverse = False
        candidates = sorted(candidates, reverse=reverse)

    winner = None

    for i, first_candidate in enumerate(candidates[:-1]):
        votes = 0
        for plist in preferences:
            for preferred_candidate in plist:
                if preferred_candidate in (first_candidate, candidates[i + 1]):
                    if preferred_candidate == first_candidate:
                        votes += 1
                    else:
                        votes -= 1
                    break
        if votes > 0:
            winner = first_candidate
        elif votes < 0:
            winner = candidates[i + 1]
        else:
            winner = first_candidate, candidates[i - 1]

    return winner


def get_candidates_from_preferences(preferences):
    candidates = []
    for voter, plist in enumerate(preferences[:-1]):
        for order, candidate in enumerate(plist):
            if candidate not in candidates:
                candidates += [candidate]
    return candidates


def condorcet_winner(preferences=None):
    """Find the winner in a set of outcome preference "votes".

    >>> A, B, C, D = 'A', 'B', 'C', 'D'
    >>> condorcet_winner(((B, C, A, D), (B, D, C, A), (D, C, A, B), (A, D, B, C), (A, D, C, B)))
    None
    """
    candidates = get_candidates_from_preferences(preferences)
    N = len(candidates)
    winner = [[] for i in range(N)]
    for i, c1 in enumerate(candidates[:-1]):
        for j, c2 in enumerate(candidates[i+1:]):
            votes = 0
            for plist in preferences:
                if index(c1) < index(c2):
                    votes += 1
                elif index(c2) < index(c2):
                    votes -= 1
            if votes >= 0:  # tie goes to first
                winner[i] += [c1]
            elif votes < 0:
                winner[i] += [c2]
    return winner


# That's a good question Gabrieli, but my understanding was that video 3 was about the paradoxical outcomes of social choice function (voting schemes) that return just a winner, not social welfare functions. The questions are about winners/choices, not rankings. But I re-listened Matt's description of pairwise elimination in lecture 2, and I can see why the accepted answer is that no pairwise elimination scheme can result in the Condorcet winner losing. Matt assumes that all alternatives are considered (in his legislature example), and ignores the problem of a tie. It is definitely silly of me to assume that a tie might be resolved with a coin flip or voter seniority. But it is nonetheless a theoretically a valid pairwise elimination voting mechanism that would fail to elect the Condorcet winner: 

# So the minimum insidious example I could come up with requires 4 voters and 4 candidates:

# preferences: (A, C, B), (B, C, A), (A, C, B), (B, C, A),
# agenda: (A, B, C)

# So we vote on A vs B and it's a tie (1 vote for each). The tie can be resolved by any number of mechanisms like voter seniority/rank/skill, candidate alphabetical order, or coin toss

# In all these "flavors" of pairwise elimination, C loses and A or B wins. And I think C is the Condorcet winner, right? So this is the paradox that subsequent lectures aim to partially resolve.