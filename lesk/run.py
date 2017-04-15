
# Importing Libraries
import argparse
from nltk.corpus import wordnet as wn

# Import Lesk Module
from lesk import Lesk

if __name__ == '__main__':

    # Checking arguments
    parser = argparse.ArgumentParser(description='Output Best Sense of Word .')
    parser.add_argument('word',
                        help='Word you want the sense of.')
    parser.add_argument('sentence',
                        help='Sentence which contains the given word.')
    args = parser.parse_args()

    sentence = args.sentence
    word = args.word

    # wordSense = wn.synsets(word)[0]
    # sentence = wordSense.examples()[0]
    # print 'Word sense', wordSense
    print 'Sentence (Input Sentence):', sentence
    print 'Word (Sense required):', word

    sense = Lesk()
    print 'Best Sense:', sense.simplifiedLesk(sentence, word=word)
