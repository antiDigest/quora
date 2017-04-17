
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
test = pd.read_csv('data/train.csv')

# train_qs = train[['id', 'question1', 'question2', 'is_duplicate']]
test_qs = test[['id', 'question1', 'question2']]

qlist = []
count = 0
for row in test_qs.itertuples():
    # print row
    try:
        if len(row[2]) > 10:
            q1 = word_tokenize(row[2].lower().decode('utf-8', errors='ignore'))
        if len(row[3]) > 10:
            q2 = word_tokenize(row[3].lower().decode('utf-8', errors='ignore'))
        qlist += q1 + q2
        count += 1
        if count % 100000 == 0:
            print 'At', count
    except TypeError or UnicodeDecodeError:
        pass

print 'Making lookup table'
qlist = list(set(qlist))
print len(qlist)
doclist = {}
count = 0
for word in qlist:
    doclist[word] = 0
    for row in test_qs.itertuples():
        if word in str(row[2]) and len(str(row[2])) > 10:
            doclist[word] += 1
        if word in str(row[3]) and len(str(row[3])) > 10:
            doclist[word] += 1
    count += 1
    if count % 10000 == 0:
        print count


print doclist
print 'All Questions added to list'
with open('data/doclist.json', 'w') as f:
    f.write(json.dumps(doclist, indent=2))
