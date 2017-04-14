
import pandas as pd
import re
from nltk import word_tokenize

import sys
sys.path.append('../.')

from lesk.lesk import Lesk


def computeOverlap(sentence1, sentence2):
    # Base
    overlap = 0

    # Step
    for word1 in sentence1:
        if word1 in sentence2:
            overlap += 1

    return overlap


def charLength(sentence):
    return len(list(sentence))


def wordLength(sentence):
    return len(word_tokenize(sentence))

if __name__ == '__main__':
    train = pd.read_csv('data/train.csv')

    train_qs = train[['id', 'question1', 'question2', 'is_duplicate']]

    sense = Lesk()

    sum1 = []
    sum2 = []
    count = 0

    for row in train_qs.itertuples():
        try:
            len1 = charLength(unicode(row[2], errors='ignore'))
            len2 = charLength(unicode(row[3], errors='ignore'))

            sum1.append(len1)
            sum2.append(len2)
            count += 1

            # if len1 < 10 or len2 < 10:
            #     print row[1], len1, len2
        except TypeError:
            # print row[1], 'TypeError'
            pass

        # try:
        #     t = wordLength(unicode(row[2], errors='ignore'))
        #     sentence1 = sense.simplifiedLesk(
        #         word_tokenize(unicode(row[2], errors='ignore')))
        #     sentence2 = sense.simplifiedLesk(
        #         word_tokenize(unicode(row[3], errors='ignore')))
        #     print str(row[1]) + ', ' + str(computeOverlap(sentence1, sentence2)) + ', ' + str(row[4])
        # except TypeError:
        #     pass

    mean1 = sum(sum1) / count
    mean2 = sum(sum2) / count
    std1 = sum([(value - mean1)**2 for value in sum1]) / count
    std2 = sum([(value - mean2)**2 for value in sum2]) / count

    print 'Mean', mean1, mean2
    print 'Variance', std1, std2
    print 'Standard Deviation', std1**(1.0 / 2.0), std2**(1.0 / 2.0)
