
import matplotlib.plot as p
import math

x = [i/500. for i in range(1000)]
f = [math.exp(-x_i) for x_i in x]
r = [x_i * f_i for x_i, f_i in zip(x, f)]
p.plot(x, r)
p.grid()
p.show()