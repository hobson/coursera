def N(k, memo={0: 0, 1: 0, 2: 1}):
	if k in memo:
		return memo[k]
	memo[k] = N(k - 2) + N(k - 3) * 2
	return memo[k]

for i in range(200):
	print i, N(i)