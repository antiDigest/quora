
import nltk
from nltk.corpus import wordnet as wn
from nltk import word_tokenize

import re
import numpy as np


class Lesk(object):

    def __init__(self, sentence):
        self.sentence = sentence
        self.meanings = {}
        for word in word_tokenize(sentence):
            self.meanings[word] = ''

    def getSenses(self, word):
        # print word
        return wn.synsets(word.lower())

    def getGloss(self, senses):

        gloss = {}

        # print senses

        for sense in senses:
            gloss[sense.name()] = []

        for sense in senses:
            gloss[sense.name()] += word_tokenize(sense.definition())

        return gloss

    def getAll(self, word):
        senses = self.getSenses(word)

        if senses == []:
            return word.lower(), senses

        return self.getGloss(senses)

    def Score(self, set1, set2):
        # Base
        overlap = 0

        # Step
        for i in range(len(set1)):
            for j in range(len(set2)):
                for word in set1[i]:
                    if word in set2[j]:
                        overlap += 1

        return overlap

    def overlapScore(self, word1, word2):

        # print word1, word2
        gloss_set1 = self.getAll(word1)
        if self.meanings[word2] == '':
            gloss_set2 = self.getAll(word2)
        else:
            gloss_set2 = self.getGloss([wn.synset(self.meanings[word2])])

        score = {}
        for i in gloss_set1.keys():
            score[i] = 0
            for j in gloss_set2.keys():
                score[i] += self.Score(gloss_set1[i], gloss_set2[j])

        # print score
        max_score = 0
        for i in gloss_set1.keys():
            if score[i] > max_score:
                max_score = score[i]
                bestSense = i

        return bestSense, max_score

    def lesk(self, word, sentence):
        maxOverlap = 0
        context = word_tokenize(sentence.lower())
        word_sense = []
        for word_context in context:
            # for sense in senses:
            if not word == word_context:
                self.meanings[word] = self.overlapScore(word, word_context)[0]

        return word, self.meanings[word], wn.synset(self.meanings[word]).definition()

# if __name__ == '__main__':

#     sentence = 'we take interest in him'
#     sense = Lesk(sentence)

#     for word in word_tokenize(sentence):
#         print sense.lesk(word, sentence)
