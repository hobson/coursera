# coursera3.py
import numpy as np

table = [[89,	7921,	96],
         [72,	5184,	74],
         [94,	8836,	87],
         [69,	4761,	78]]
x1 = np.array([float(r[0]) for r in table])
x2 = np.array([r[1] for r in table])
y = np.array([r[2] for r in table])

nx1 = (x1 - np.mean(x1)) * 1.0 / float(max(x1) - min(x1))
nx2 = (x2 - np.mean(x2)) * 1.0 / float(max(x2) - min(x2))

print 'normalized x1'
print nx1 
print 'normalized x2'

print nx2
ans1 = nx2[1]

print 'ans1: %s' % repr(ans1)

#print 'ans2: %f' % ans2

#print 'ans3: %f' % ans3

