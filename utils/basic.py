
import nltk
from nltk import word_tokenize
import pandas as pd


def tokenize(q1, q2):
    """
        q1 and q2 are sentences/questions. Function returns a list of tokens for both.
    """
    return word_tokenize(q1), word_tokenize(q2)


def posTag(q1, q2):
    """
        q1 and q2 are lists. Function returns a list of POS tagged tokens for both.
    """
    return nltk.pos_tag(q1), nltk.pos_tag(q2)


def stemmer(tag_q1, tag_q2):
    """
        tag_q = tagged lists. Function returns a stemmed list.
    """

    stem_q1 = []
    stem_q2 = []

    for token in tag_q1:
        stem_q1.append(stem(token))

    for token in tag_q2:
        stem_q2.append(stem(token))

    return stem_q1, stem_q2
