
from nltk import word_tokenize
import pandas as pd
from collections import Counter

import sys
sys.path.append('../.')
from utils.similarity import cosine
from utils.tfidf import tfidf

reload(sys)
sys.setdefaultencoding('utf-8')

train = pd.read_csv('../quora/data/train.csv')
test = pd.read_csv('../quora/data/test.csv')

train_qs = train[['id', 'question1', 'question2', 'is_duplicate']]
test_qs = test[['test_id', 'question1', 'question2']]

qlist = []
for row in test_qs.itertuples():
    # print row
    try:
        if len(row[2]) > 10:
            qlist.append(word_tokenize(row[2].decode(
                'utf8', errors='ignore').lower()))
        if len(row[3]) > 10:
            qlist.append(word_tokenize(row[3].decode(
                'utf8', errors='ignore').lower()))
    except TypeError or UnicodeDecodeError:
        pass

print 'Making lookup table'
qlist = dict(Counter(qlist))

# print qlist

# print len(qlist)
print 'All Questions added to list'

for row in test_qs.itertuples():
    wordvec1 = word_tokenize(row[2].lower())
    wordvec2 = word_tokenize(row[3].lower())
    words = wordvec1 + wordvec2
    words = list(set([word for word in words if word != '?']))

    # print words

    vec1 = []
    vec2 = []
    for word in words:
        vec1.append(tfidf(wordvec1, qlist, word))
        vec2.append(tfidf(wordvec2, qlist, word))

    with open('data/submission.csv', 'a') as f:
        f.write(str(row[1]) + "," + str(cosine(vec1, vec2)) + '\n')

    print str(row[1]) + "," + str(cosine(vec1, vec2))
