from numpy import *

def matrix_distance(a, b):
    d = 0
    for i in range(shape(a)[0]):
        for j in range(shape(a)[1]):
            dif = a[i, j] - b[i, j]
            d += dif * dif
    return d


def factorize(v, Np=10, N_iter=50):
    Ni, Nf = shape(v)
    w = matrix([[random.random() for j in range(Np)] for i in range(Ni)])
    h = matrix([[random.random() for i in range(Nf)] for i in range(Np)])

    for i in range(N_iter):
        wh = w * h
        cost = matrix_distance(v, wh)
        if not i % 10:
            print cost

        if not cost:
            break

        w_trans = transpose(w)
        hn = w_trans * v
        hd = w_trans * wh

        h = matrix(array(h) * array(hn) / array(hd))

        h_trans = transpose(h)
        wn = v * h_trans
        wd = w * h * h_trans

        w = matrix(array(w) * array(wn) / array(wd))

    return w, h