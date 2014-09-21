# coursera3.py
import numpy as np


def normalize(X):
    """Normalize each column to range between -0.5 and +0.5 (approximately).
    
    X_normalized = X - mean(X) / (max(X) - min(X)
    
    TODO: A faster approach may be: 0.5 * (X - min(X)) / (max(X) - min(X))

    >>> table = [[89,	7921,	96], [72,	5184,	74], [94,	8836,	87], [69,	4761,	78]]
    >>> normalize(table)  # doctest: +NORMALIZE_WHITESPACE
    matrix([[ 0.32      ,  0.30564417,  0.55681818],
            [-0.36      , -0.36601227, -0.44318182],
            [ 0.52      ,  0.53018405,  0.14772727],
            [-0.48      , -0.46981595, -0.26136364]])
    """
    # columnwise
    X = np.matrix(X)
    Xt = X.T
    mu = X.mean(0)
    Xmin = X.min(0)
    Xmax = X.max(0)
    return (X - mu) / (Xmax - Xmin)


def regression_cost(X, y, theta):
    """Compute the cost function for polynomial regression.
    
    J = 1 / len(y) * sum((y - X * theta) ** 2)
    
    >>> X = [[89,	7921], [72,	5184], [94,	8836], [69,	4761]]
    >>> y = [96, 74, 87, 78]
    >>> regression_cost(X, y, [1.0, -0.01])  # doctest: +ELLIPSIS
    5038.7628500...
    """
    yT = np.matrix(y).T
    thetaT = np.matrix(theta).T
    X = np.matrix(X)
    return (sum(np.power(yT - X * thetaT, 2)) / len(y))[0,0]
