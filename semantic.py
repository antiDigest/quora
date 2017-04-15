import nltk
from nltk import word_tokenize
import pandas as pd


from stemming.porter import stem

from utils.tfidf import tfidf


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


def semanticSimilarity(q1, q2):

    tokens_q1, tokens_q2 = tokenize(q1, q2)
    # stem_q1, stem_q2 = stemmer(tokens_q1, tokens_q2)
    tag_q1, tag_q2 = posTag(tokens_q1, tokens_q2)

    for i, word1 in enumerate(tag_q1):
        for j, word2 in enumerate(tag_q2):
            if word1[1] == word2[1] and ('NN' in word1 or 'VB' in word1):
                print word1, word2

if __name__ == '__main__':
    train = pd.read_csv('data/train.csv')

    train_qs = train[['id', 'question1', 'question2', 'is_duplicate']]

    # sense = Lesk()

    sum1 = []
    sum2 = []
    count = 0

    for row in train_qs.itertuples():
        try:
            count += 1
            q1 = row[2].decode('utf8', errors='ignore')
            q2 = row[3].decode('utf8', errors='ignore')
            semanticSimilarity(q1, q2)
            # break
        except TypeError:
            pass
