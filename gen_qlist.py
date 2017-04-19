
from nltk import word_tokenize
import pandas as pd
import numpy as np
import nltk
import re

from sklearn.metrics.pairwise import linear_kernel

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import json

from utils.similarity import cosine

import sys

STOP_WORDS = nltk.corpus.stopwords.words()


def clean_sentence(val):
    "remove chars that are not letters or numbers, downcase, then remove stop words"
    regex = re.compile('([^\s\w]|_)+')
    sentence = regex.sub('', val).lower()
    sentence = sentence.split(" ")

    for word in list(sentence):
        if word in STOP_WORDS:
            sentence.remove(word)

    sentence = " ".join(sentence)
    return sentence


def fit_tfs(X):
    tfidf = TfidfVectorizer(tokenizer=word_tokenize)

    tdm = tfidf.fit_transform(X)

    print tdm.shape
    # print tdm.stop_words_

    return tdm

data = pd.read_csv('data/train.csv')
data = data.dropna(how="any")
X_set1 = data[data['question1'].str.len() >= 10]['question1']
X_set2 = data[data['question2'].str.len() >= 10]['question2']
y = data['is_duplicate']
X = X_set1.append(X_set2, ignore_index=True)

print 'Cleaning Data, this might take some time.'
for col in ['question1', 'question2']:
    data[col] = data[col].apply(clean_sentence)

data.to_csv('data/cleaned_train.csv')


# print 'Making Term-Document Matrix'
# tdm = fit_tfs(X).toarray()

# # print data.shape

# # print 'Calculating similarities'
# # cosine_similarities = cosine_similarity(
# #     tdm[:data.shape[0]], tdm[data.shape[0]:])

# print 'Lets check similarity'
# # with open('data/submit.csv', 'w') as f:
# for i in range(data.shape[0]):
#     print i, y[i], cosine(tdm[i], tdm[data.shape[0] + i - 1])
#     # f.write(data['test_id'][i] + ',' + cosine_similarities[i] + '\n')

# # print 'Calculating first cosine similarity'
# # cosine_similarities = linear_kernel(tfidf, tfidf[404290]).flatten()

# # print cosine_similarities
