
from nltk import word_tokenize
import pandas as pd
import numpy as np

from sklearn.metrics.pairwise import linear_kernel

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import json

from utils.similarity import cosine

import sys


def fit_tfs(X):
    tfidf = TfidfVectorizer(tokenizer=word_tokenize)

    tdm = tfidf.fit_transform(X)

    # print tdm.shape

    return tdm

data = pd.read_csv('data/test.csv')
X_set1 = data[data['question1'].str.len() >= 10]['question1']
X_set2 = data[data['question2'].str.len() >= 10]['question2']

X = X_set1.append(X_set2, ignore_index=True)

print 'Making Term-Document Matrix'
tdm = fit_tfs(X).toarray()

print 'Lets check similarity'
with open('data/submit.csv', 'w') as f:
    for i in range(data.shape[0]):
        print data['test_id'][i], cosine(tdm[i], tdm[data.shape[0] + i - 1])
        f.write(data['test_id'][i] + ',' + cosine_similarities[i] + '\n')
