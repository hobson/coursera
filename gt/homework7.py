c1 = Coalition([0, 0, 0, .8, .8, .8, 1], N=3)
%run Coalition
c1 = Coalition([0, 0, 0, .8, .8, .8, 1], N=3)
c1.shapley_values
c1.shapley_values()
hw2 = Coalition([0, 0,0, 0, 1,1,1],3)
hw2
hw2.shapley_values()
sv2 = hw2.shapley_values()
hw3 = Coalition([0, 0, 0, 0, 0, 1,1,1,1,1,1,1, 2],3)
sv3 = hw3.shapley_values()
sv3
hw5 = Coalition([0, 0, 0, 4, 4, 4, 6],3)
hw5
hw5.shapley_value()
hw5.shapley_values()
hw7 = Coalition([0, 0, 0, 4, 4, 0, 6],3)
hw7.shapley_values()
sum(hw7.shapley_values())
sv7 = hw7.shapley_values()
sv7
[4*v/sum(sv7) for v in sv7]
7/3.
[4*v/sum(sv7) for v in sv7]
sv7
hw7
%run Coalition
hw7 = Coalition([0, 0, 0, 4, 4, 0, 6],3)
sv7 = hw7.shapley_values()
sv7
[4*v/sum(sv7) for v in sv7]
[4*v/sum(sv7) for v in sv7]
[3*v/sum(sv7) for v in sv7]
parliament = Coalition([45, 25, 15, 15, 45+25, 45+15, 45+15, 25+15, 25+15, 30, 45+25+15, 45+25+15, 45+15+15, 25+15+15, 45+25+15+15], N=4)

