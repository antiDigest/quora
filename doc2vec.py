
from nltk import word_tokenize
import pandas as pd
import numpy as np
import nltk
import re

from sklearn.metrics.pairwise import linear_kernel

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter

from sklearn.model_selection import train_test_split as tts

from gensim.models.doc2vec import Doc2Vec, LabeledSentence
import gensim

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


def concatenate(data):
    X_set1 = data['question1']
    X_set2 = data['question2']
    X = X_set1.append(X_set2, ignore_index=True)

    return X

X_train = pd.read_csv('data/cleaned_train.csv')
# X_train = X_train.dropna(how="any")

# y = X_train['is_duplicate']

print 'Exported Cleaned test Data, no need for cleaning'
# for col in ['question1', 'question2']:
#     X_train[col] = X_train[col].apply(clean_sentence)

# print 'Getting the test sets.'
# X_test = pd.read_csv('data/cleaned_test.csv')
# X_test = X_test.dropna(how="any")

# print 'Cleaning test Data, this might take time'
# for col in ['question1', 'question2']:
#     X_test[col] = X_test[col].apply(clean_sentence)

# print('Exporting cleaned test set to a new file for easy reference')
# X_test.to_csv('data/cleaned_test.csv', index=False)

print 'Now training Doc2vec over test documents'
from gensim.models.doc2vec import Doc2Vec
from gensim.models import doc2vec


class LabeledLineSentence(object):

    def __init__(self, doc_list, labels_list):
        self.labels_list = labels_list
        self.doc_list = doc_list

    def __iter__(self):
        for idx, doc in enumerate(self.doc_list):
            yield doc2vec.TaggedDocument(words=word_tokenize(str(doc)), tags=[self.labels_list[idx]])

import multiprocessing
cores = multiprocessing.cpu_count()
print 'How many cores does a mac have ?', cores
assert gensim.models.doc2vec.FAST_VERSION > -1

print('Concatenating train data')
X = concatenate(X_train)
labels = []
for label in X_train['id'].tolist():
    labels.append('SENT_%s_1' % label)
for label in X_train['id'].tolist():
    labels.append('SENT_%s_2' % label)

print('Training doc2vec model over the train data.')
docs = LabeledLineSentence(X.tolist(), labels)
it = docs.__iter__()
model1 = Doc2Vec(size=50, window=4, min_count=0, workers=3)
model1.build_vocab(it)

for epoch in range(10):
    model1.train(it, total_examples=len(labels), epochs=1)
    model1.alpha -= 0.0002  # decrease the learning rate
    model1.min_alpha = model1.alpha  # fix the learning rate, no decay
    model1.train(it, total_examples=len(labels), epochs=1)

print 'Getting question lists'
X_train_qs_1 = X_train['question1'].tolist()
X_train_qs_2 = X_train['question2'].tolist()
is_duplicate = X_train['is_duplicate'].tolist()

print 'Printing doc2vec output on test documents.'
# X_train.index = np.arange(0, X_test.shape[0])
# print(X_test)

count = 0
with open('data/submit_train.csv', 'w') as f:
    f.write('id,is_duplicate\n')
    for i in X_train['id'].tolist():
        doc1 = word_tokenize(str(X_train_qs_1[count]))
        doc2 = word_tokenize(str(X_train_qs_2[count]))

        docvec1 = model1.infer_vector(doc1)
        docvec2 = model1.infer_vector(doc2)

        sim = cosine(docvec1, docvec2)
        count += 1
        if count % 10000 == 0:
            print(str(count) + ", " + str(i) + ', ' +
                  str(sim) + ', ' + str(is_duplicate[count]))
        f.write(str(i) + ',' + str(sim) + '\n')
        # break
