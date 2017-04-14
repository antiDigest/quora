# Information on files:


## overlap.py

*	Runs over the sentences to find out the best possible sense of the sentence and find the most overlap among the two sentences using the senses.

*	


## Some Quirks:

*	Some of the question lengths are less than 10 characters, those pairs have been left out.

*	*q1id* and *q2id* have been left out because they don't indicate anything, plus for the same question at different places, they are different.

*	Some labels do not seem true, especially for the duplicate ones. I decided to rely on the labels and defer pruning due to hard manual effort.


# More NLP (not Deep Learning) methods:

*	Semantic Relations - Find Semantic similarity between two sentences.

*	Lexical Chains

*	Entailment using Logic Representation of text (Extremely hard)


# TODO:

1.	Find semantic similarity between two sentences.

2.	Implement lexical chains