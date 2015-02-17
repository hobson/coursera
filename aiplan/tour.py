#tour.py

h = lambda x:x
h.A=366
h.B=0
h.C=160
h.D=242
h.E=161
h.F=176
h.G=77
h.H=151
h.L=244
h.M=241
h.N=234
h.O=380
h.P=100
h.S = 253 
h.R=193            
L = h.L+111              # 355
A = h.A+118              # 484
M = h.M + 70 + 111       # 422
S = 118 + 140 + h.S      # 511
D = 111 + 70 + 75 + h.D  # 498
C = 111+70+75+120+h.C    # 536
F = 118+140+99+h.F       # 533
R = 118+140+80+h.R       # 531
P = 118+140+80+97+h.P    # 535
B_throughSF = 211+99+140+118   # 568
B_throughR = 101+97+80+140+118 # 536
P_throughC = 614               # 614
B_throughPC = P_throughC + 101 # 715

# textbook example starting at A
seq = 'ASFRPB'

# so probably
seq = 'TLASFRPB'

# these are wrong
seq = 'TLAMDSRFPB'
seq = 'TLAMDSRFCPB'

d = lambda x:x

def h_tour(city):
    return getattr(d, city) + getattr(h, city)