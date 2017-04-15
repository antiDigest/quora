#ReadMe.txt

--------------------------------------------------------------RUN---------------------------------------------------------------

Command to run:

		"python run.py <word> <sentence>"

<word> and <sentence> are required arguments.

<word>: The word you require the sense to be found for.

<sentence>: The sentence to look into (should contain the word).


-------------------------------------------------------POSITIVE EXAMPLES--------------------------------------------------------

1.
Sentence (Input Sentence): all work and no play makes jack a dull boy

Word (Sense required): boy

Best Sense: (Synset('male_child.n.01'), u'a youthful male person')

2.
Sentence (Input Sentence): they had all kinds of books in the library

Word (Sense required): library

Best Sense: (Synset('library.n.01'), u'a room where books are kept')


--------------------------------------------------------NEGATIVE EXAMPLE--------------------------------------------------------

Sentence (Input Sentence): to work inside his house was a terrible idea

Word (Sense required): house

Best Sense: (Synset('family.n.01'), u'a social unit living together')