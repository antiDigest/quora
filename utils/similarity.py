import math
import numpy as np
from scipy import spatial


def cosine(v1, v2):
    """
            v1 and v2 are two vectors (can be list of numbers) of the same dimensions. Function returns the cosine distance between those
            which is the ratio of the dot product of the vectors over their RS.
    """
    v1 = np.array(v1)
    v2 = np.array(v2)

    return np.dot(v1, v2) / (np.sqrt(np.sum(v1**2)) * np.sqrt(np.sum(v2**2)))

if __name__ == '__main__':
    from random import randint
    v1 = [randint(500, 1000) for i in range(1, 10)]
    v2 = [randint(1, 10) for i in range(1, randint(1, 10))]
    result = 1 - spatial.distance.cosine(v1, v2)
    print v1, v2
    print cosine(v1, v2)
    print result
