
from nltk import word_tokenize
import pandas as pd
from collections import Counter
import json

from utils.similarity import cosine
from utils.tfidf import tfidf

import sys
reload(sys)
sys.setdefaultencoding('utf-8')

print 'Into the loop I go'
# train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

# train_qs = train[['id', 'question1', 'question2', 'is_duplicate']]
test_qs = test[['test_id', 'question1', 'question2']]

# qlist = []
# count = 0
# for row in test_qs.itertuples():
#     # print row
#     try:
#         if len(row[2]) > 10:
#             q1 = word_tokenize(row[2].lower().decode('utf-8', errors='ignore'))
#         if len(row[3]) > 10:
#             q2 = word_tokenize(row[3].lower().decode('utf-8', errors='ignore'))
#         qlist += q1 + q2
#         count += 1
#         if count % 10000 == 0:
#             print 'At', count
#     except TypeError or UnicodeDecodeError:
#         pass

# print 'Making lookup table'
# qlist = dict(Counter(qlist))
# print qlist
# print 'All Questions added to list'
# with open('data/qlist.json', 'w') as f:
#     f.write(json.dumps(qlist, indent=2))

# with open('submission.csv', 'a') as f:
#     f.write('id,is_duplicate\n')

import json
qlist = json.loads(open('data/qlist.json').read())
# print qlist['ciptv1']

print 'now starting'
with open('submission.csv', 'a') as f:
    count = 0
    for row in test_qs[121423:].itertuples():
        if len(str(row[2])) > 10 and len(str(row[3])) > 10:
            wordvec1 = word_tokenize(
                row[2].lower().decode('utf-8', errors='ignore'))
            wordvec2 = word_tokenize(
                row[3].lower().decode('utf-8', errors='ignore'))
            words = wordvec1 + wordvec2
            words = list(set([word for word in words if word != '?']))

            # print words

            vec1 = []
            vec2 = []
            for word in words:
                vec1.append(tfidf(wordvec1, qlist, word))
                vec2.append(tfidf(wordvec2, qlist, word))

            f.write(str(row[1]) + "," + str(cosine(vec1, vec2)) + '\n')
        else:
            f.write(str(row[1]) + "," + '0' + '\n')

        count += 1
        if count % 10000 == 0:
            print str(row[1]) + "," + str(cosine(vec1, vec2))
