"""
Review Section 2 of Coursera Machine Learning Class
"""

import numplot as plt
import numpy as np

x = [5, 3, 0, 4]
y = [4, 4, 1, 3]
A = np.vstack([np.ones(len(x)), np.array(x)]).T
J = np.array([0, 1])
h = np.dot(A, J)
ans1 = m = len(x)
ans2 = J01 = 0.5 / m * sum((h.T - y)**2)
J = np.array([-1, 0.5])
ans3 = h = np.dot(A, J)[3]

print 'ans1: %f' % ans1

print 'ans2: %f' % ans2

print 'ans3: %f' % ans3

print plt.regressionplot(x, y)
