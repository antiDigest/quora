import numpy as np


def min(a, b, c):
    m = a if a < b else b
    m = m if m < c else c
    return m


def MinDist(n, m, mind, x, y):
    for i in xrange(m + 1):
        mind[0, i] = i

    for i in xrange(n + 1):
        mind[i, 0] = i

    for i in xrange(1, n + 1):
        for j in xrange(1, m + 1):

            a = mind[i - 1, j] + 1
            b = mind[i, j - 1] + 1

            if x[i - 1] == y[j - 1]:
                c = mind[i - 1, j - 1]
            else:
                c = mind[i - 1, j - 1] + 2

            mind[i, j] = min(a, b, c)

    return mind[n, m]


def dist(x, y):

    mind = np.ndarray(shape=(len(x) + 1, len(y) + 1), dtype=int)

    n = len(x)
    m = len(y)

    k = MinDist(n, m, mind, x, y)

    return k
