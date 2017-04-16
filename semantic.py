import nltk
from nltk import word_tokenize
import pandas as pd
from nltk.corpus import wordnet as wn

import json

from stemming.porter import stem
from lesk.lesk import Lesk
from utils.tfidf import tfidf
from utils.similarity import path
from utils.basic import tokenize, posTag, stemmer


def computePath(q1, q2):
    wordsim = {}
    # maxsim = 0.0
    for word1 in q1:
        wordsim[word1[0]] = None
        maxsim = 0.0
        for word2 in q2:
            # print word1[1], word2[1]
            sim = path(wn.synset(word1[1]), wn.synset(word2[1]))
            # print 'Path similarity', sim
            if sim > maxsim and sim != None:
                maxsim = sim
                wordsim[word1[0]] = (word2[0], sim)

    print json.dumps(wordsim, indent=2)


def semanticSimilarity(q1, q2):

    tokens_q1, tokens_q2 = tokenize(q1, q2)
    # stem_q1, stem_q2 = stemmer(tokens_q1, tokens_q2)
    tag_q1, tag_q2 = posTag(tokens_q1, tokens_q2)

    sentence = []
    for i, word in enumerate(tag_q1):
        if 'NN' in word[1] or 'JJ' in word[1] or 'VB' in word[1]:
            sentence.append(word[0])

    sense1 = Lesk(sentence)
    sentence1Means = []
    for word in sentence:
        sentence1Means.append(sense1.lesk(word, sentence))

    sentence = []
    for i, word in enumerate(tag_q2):
        if 'NN' in word[1] or 'JJ' in word[1] or 'VB' in word[1]:
            sentence.append(word[0])

    sense2 = Lesk(sentence)
    sentence2Means = []
    for word in sentence:
        sentence2Means.append(sense2.lesk(word, sentence))

    # for i, word in enumerate(sentence1Means):
    #     print sentence1Means[i][0], sentence2Means[i][0]

    return computePath(sentence1Means, sentence2Means)

if __name__ == '__main__':
    train = pd.read_csv('data/train.csv')

    train_qs = train[['id', 'question1', 'question2', 'is_duplicate']]

    # sense = Lesk()

    sum1 = []
    sum2 = []
    count = 0

    for row in train_qs.itertuples():
        q1 = row[2].decode('utf8', errors='ignore')
        q2 = row[3].decode('utf8', errors='ignore')
        # print q1, q2
        semanticSimilarity(q1, q2)
        break
