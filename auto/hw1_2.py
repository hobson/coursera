def N(k, memo={0: 0, 1: 0, 2: 2, 3: 0}):
    if k in memo:
        return memo[k]
    memo[k] = N(k - 2, memo) + 2 * N(k - 3, memo)
    return memo[k]


import sys

if __name__ == '__main__':
    try:
        num = int(sys.argv[1])
    except:
        num = 10
    print N(num)
