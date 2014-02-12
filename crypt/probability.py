#probability

def key_combinations(symbols=2, length=128):
    return symbols ** length

def p_guess(symbols=2, length=128):
    return 1 / float(key_combinations(symbols, length))

def t_exhaustive(hash_rate, symbols, length):
    """
    >>> t_exhaustive(1e12 * 1.e9 / 200., symbols=2, length=128)
    """
    return key_combinations(symbols, length) / hash_rate