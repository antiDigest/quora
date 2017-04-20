import nltk
from nltk import word_tokenize
import pandas as pd
from nltk.corpus import wordnet as wn

import numpy as np
import json

from stemming.porter import stem
from lesk.lesk import Lesk
from utils.tfidf import tfidf
from utils.similarity import path, wup, edit
from utils.basic import tokenize, posTag, stemmer

from sklearn.metrics import log_loss


def computeWup(q1, q2):

    R = np.zeros((len(q1), len(q2)))

    for i in range(len(q1)):
        for j in range(len(q2)):
            if q1[i][1] == None or q2[j][1] == None:
                sim = edit(q1[i][0], q2[j][0])
            else:
                sim = wup(wn.synset(q1[i][1]), wn.synset(q2[j][1]))

            if sim == None:
                sim = edit(q1[i][0], q2[j][0])

            R[i, j] = sim

    # print R

    return R


def computePath(q1, q2):

    R = np.zeros((len(q1), len(q2)))

    for i in range(len(q1)):
        for j in range(len(q2)):
            if q1[i][1] == None or q2[j][1] == None:
                sim = edit(q1[i][0], q2[j][0])
            else:
                sim = path(wn.synset(q1[i][1]), wn.synset(q2[j][1]))

            if sim == None:
                sim = edit(q1[i][0], q2[j][0])

            R[i, j] = sim

    # print R

    return R


def overallSim(q1, q2, R):

    sum_X = 0.0
    sum_Y = 0.0

    for i in range(len(q1)):
        max_i = 0.0
        for j in range(len(q2)):
            if R[i, j] > max_i:
                max_i = R[i, j]
        sum_X += max_i

    for i in range(len(q1)):
        max_j = 0.0
        for j in range(len(q2)):
            if R[i, j] > max_j:
                max_j = R[i, j]
        sum_Y += max_j

    if (float(len(q1)) + float(len(q2))) == 0.0:
        return 0.0

    overall = (sum_X + sum_Y) / (2 * (float(len(q1)) + float(len(q2))))

    return overall


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
    #     print sentence1Means[i][1], sentence2Means[i][1]

    R1 = computePath(sentence1Means, sentence2Means)
    R2 = computeWup(sentence1Means, sentence2Means)

    R = (R1 + R2) / 2

    return overallSim(sentence1Means, sentence2Means, R)

if __name__ == '__main__':
    train = pd.read_csv('data/cleaned_train.csv')

    train_qs = train[['id', 'question1', 'question2', 'is_duplicate']]
    y_train = train['is_duplicate']

    y_pred = []
    count = 0
    print('Calculating similarity for the training data, please wait.')

    for row in train_qs.itertuples():
        # print row
        q1 = str(row[2]).decode('utf8', errors='ignore')
        q2 = str(row[3]).decode('utf8', errors='ignore')

        sim = semanticSimilarity(q1, q2)
        count += 1
        if count % 10000 == 0:
            print count, sim, row[4]
        y_pred.append(sim)

    output = pd.DataFrame(list(zip(train_qs['id'], y_pred)), columns=[
                          'id', 'similarity'])

    output.to_csv('data/semantic_train.csv')

    print log_loss(y_train, np.array(y_pred))
