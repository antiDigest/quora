
import nltk
from nltk.corpus import wordnet as wn
from nltk import word_tokenize
from nltk.corpus.reader import NOUN

import re
import numpy as np


class Lesk(object):

    def __init__(self):
        pass

    def getMostFrequent(self, word):
        senses = wn.synsets(word.lower())
        # print word.lower(), senses
        if senses == []:
            return word.lower(), senses
        maxValSense = len(senses[0].examples())
        freqSense = senses[0]
        for sense in senses:
            # print len(sense.examples())
            if len(sense.examples()) > maxValSense:
                maxValSense = len(sense.examples())
                freqSense = sense

        return freqSense, senses

    def getAll(self, word):
        senses = wn.synsets(word.lower())

        if senses == []:
            return word.lower(), senses

        gloss = {}

        for sense in senses:
            gloss[sense.name()] = []

        for sense in senses:
            gloss[sense.name()] += word_tokenize(sense.definition())

        # print gloss

        return gloss

    def computeOverlap(self, signature, context):

        # Base
        overlap = 0

        # Step
        for word in context:
            if word in signature:
                overlap += 1

        return overlap

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
        nyms = ['hypernyms', 'hyponyms',
                'member_meronyms', 'part_meronyms', 'substance_meronyms']

        gloss_set1 = self.getAll(word1)
        gloss_set2 = self.getAll(word2)

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

        return bestSense, wn.synset(bestSense).definition()

    def adaptedLesk(self, sentence):
        maxOverlap = 0
        context = word_tokenize(sentence.lower())

        # index = context.index(word.lower())
        # if index - k > 0 and index + k < len(context):
        #     context = context[index - k: index + k]
        # elif index + k < len(context):
        #     context = context[0: index + k]
        # elif index - k > 0:
        #     context = context[index - k: len(context)]
        # else:
        #     context = context[0: len(context)]

        print context
        new_sent = []
        for i, word in enumerate(context):
            if i >= 0 and i < len(context) - 1:
                new_sent.append(
                    (word, self.overlapScore(word, context[i + 1])))
            elif i == len(context) - 1:
                new_sent.append(
                    (word, self.overlapScore(word, context[i - 1])))

                # for sense in senses:

        return new_sent

    def simplifiedLesk(self, sentence, word):

        bestSense, senses = self.getMostFrequent(word)
        maxOverlap = 0
        context = word_tokenize(sentence.lower())

        if senses == []:
            return 'ERROR: Cannot find a proper sense for the word !'

        for sense in senses:
            examples = [sense.definition()] + sense.examples()
            signature = []
            for example in examples:
                signature += list(
                    set(re.split('[^\w]* [^\w]*', example.lower())))
            signature = list(set(signature))
            # print signature
            overlap = self.computeOverlap(signature, context)
            if overlap > maxOverlap:
                maxOverlap = overlap
                bestSense = sense

        return bestSense, bestSense.definition()

if __name__ == '__main__':
    sense = Lesk()

    print sense.adaptedLesk('pine cone')
