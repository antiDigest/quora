from stemming.porter2 import stem
from textblob import TextBlob
from minEditDist import dist
import sys
import re
reload(sys)
sys.setdefaultencoding('utf-8')
import nltk.data


def find_sentence(var):
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    fp = open("file1.txt")
    data = fp.read().decode("utf-8-sig")
    sentence = tokenizer.tokenize(data)
    for sent in sentence:
        a = sent.split()
        for word in a:
            if var == word:
                return sent
    return -1


def check_word(var):
    a = TextBlob(var)
    new = a.correct()

    if var == new:
        #     print "Word = '"+var+"'"
        #     x = var
        #     nearlist = []
        #     with open("aux/bigger.txt") as myfile:
        #         for line in myfile:
        #             y= line.strip()
        #             k = dist(x,y)
        #             if k == 1 or k==2:
        #                 nearlist += [y]
        #             else:
        #                 ind =  words.index(x)
        return 1
        # print "Current word : ",words[ind]
        # print "Previous Word : ",words[ind-1]
        # print "Next Word : ",words[ind+1]

        # sent = find_sentence(var)
        # print sent

    else:
        # x = var
        # nearlist = []
        # with open("aux/bigger.txt") as myfile:
        #     for line in myfile:
        #         y = line.strip()
        #         k = dist(x,y)
        #         if k == 1 or k==2:
        #             nearlist += [y]
        #         else:
        #             ind =  words.index(x)
        return -1


def check_spelling(var):
    with open("dictionary.txt") as myfile:
        for line in myfile:
            if line.strip() == var.lower():
                return 1
    return -1


def check(var):

    afs = stem(var)
    flag = check_spelling(afs.strip())

    if flag == -1:
        return check_word(var)


def count_errors(text):
    # file1 = open('file1.txt', 'r')
    # xyz = file1.read().decode("utf-8-sig")
    text = text.decode("utf-8-sig")
    global words
    words = re.split('[\s?\.!,:; ]+', text)[:-1]

    errors = 0

    for word in words:

        flag = check_spelling(word.strip())

        if flag == -1:
            if check_word(word) == -1:
                errors += 1

    # print errors
    return errors

# if __name__ == "__main__":
#     main()
#     # find_sentence()
