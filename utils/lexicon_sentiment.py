# Sentiment extract

from textblob import TextBlob
import glob
import random
import re
import string
import csv
import nltk
from nltk.corpus import stopwords, wordnet
from stemming.porter2 import stem
import math
import sys
import os
import codecs
reload(sys)
sys.setdefaultencoding('utf-8')

parenthesis = [')', '(', ']', '[', '{', '}', '*', '&', '\\',
               '!', '$', '^', ';', '<', '>', '?', '_', '=', '+', 'RT', '.']
customstopwords = ['band', 'they', 'them', 'and', 'the']

pos = 'aux/positive.txt'
neg = 'aux/negative.txt'

with open(pos, 'r') as f:
    positive = f.read().split('\n')

with open(neg, 'r') as f:
    negative = f.read().split('\n')

# print stopwords.words('english')


def getTaggedWords():

    neglist = []
    poslist = []

    for i in range(0, len(negative)):
        neglist.append(-1)

    for i in range(0, len(positive)):
        poslist.append(1)

    postagged = zip(positive, poslist)
    negtagged = zip(negative, neglist)

    taggedtext = postagged + negtagged

    return taggedtext


def getWords(taggedtext):  # seems correct
    # print wordlist

    tweets = []
    customstopwords = ['band', 'they', 'them']

    for (word, sentiment) in taggedtext:
        word_filter = [i.lower() for i in word.split()]
        tweets.append((word_filter, sentiment))

    wordlist = getWordFeatures(getAllWords(tweets))
    wordlist = [i for i in wordlist if not i in stopwords.words('english')]
    wordlist = [i for i in wordlist if not i in customstopwords]

    return wordlist


def textCleaner(value):
    # print value

    for i in parenthesis:
        value = value.replace(i, '')

    for i in string.punctuation:
        value = value.replace(i, '')
    return value


def getAllWords(tweets):
    allwords = []
    for (words, sentiment) in tweets:
        allwords.extend(words)
    return allwords

# Order a list of tweets by their frequency.


def getWordFeatures(listoftweets):
    # Print out wordfreq if you want to have a look at the individual counts
    # of words.
    wordfreq = nltk.FreqDist(listoftweets)
    words = wordfreq.keys()
    return words


def makeDocument(tweets):  # seems correct
    documents = []
    for (words, sentiment) in tweets:
        words_filtered = [e.lower() for e in words]
        documents.append((words_filtered, sentiment))
    return documents


def textClean(s):
    remove = ['\\t', '\\n', '  ']
    for i in remove:
        s = re.sub(i, '', s)
    s = s.lower()
    s = s.split()
    return s

taggedtext = getTaggedWords()
word_features = getWords(taggedtext)


def sentiment(text):

    # raw = re.split('[\s?\.!,:; ]+', text)[:-1]

    document_words = set(textClean(text))

    value = 0

    for word in document_words:
        if word in positive:
            value += 1
        elif word in negative:
            value -= 1

    return value


def Polarity(text):
    b = text.decode('utf-8-sig')
    text = TextBlob(b)
    polarity = text.sentiment.polarity
    return polarity
