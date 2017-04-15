
import math
from collections import Counter


def tf(question, word):
    if word not in question:
        return 0
    count = dict(Counter(question))
    q_len = len(question)
    return float(count[word]) / float(q_len)


def n_containing(qlist, word):
    return float(qlist[word])


def idf(qlist, word):
    # print len(qlist.keys()), (1 + n_containing(qlist, word))
    return math.log(float(len(qlist.keys())) / (1.0 + n_containing(qlist, word)))


def tfidf(question, qlist, word):
    return tf(question, word) * idf(qlist, word)
