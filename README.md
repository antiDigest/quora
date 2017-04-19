# Information on files:

# Methods Implemented

### Cosine Similarity using TF-IDF Weighting

*	Calculating TF-IDF vector for each sentence and finding the cosine similarity measure for two TF-IDF vectors calculated from the question pairs.

### Semantic Similarity Using Adapted Lesk algorithm:

*	The *adapted Lesk Algorithm* uses context instead of just one single word.
The context is defined as k words around the word.
All the senses of all the words around the given word identify their glosses.
These glosses are compared with different glosses of all the senses for that particular word.
The overall score for the word is calculated using a method (as given in the SLP 2nd Edition)
The best sense is the one with the max overall score.

*	

## Some Quirks:

*	Some of the question lengths are less than 10 characters, those pairs have been left out.

*	*q1id* and *q2id* have been left out because they don't indicate anything, plus for the same question at different places, they are different.

*	Some labels do not seem true, especially for the duplicate ones. I decided to rely on the labels and defer pruning due to hard manual effort.


# More NLP (not Deep Learning) methods:

*	Semantic Relations - Find Semantic similarity between two sentences.

*	Lexical Chains

*	Entailment using Logic Representation of text (Extremely hard)


## Options (research found):

1.	Semantic similarity using Maximal Weighted Bipartite Graph Matching

2.	Find semantic similarity between two sentences.

3.	From Word Embeddings to Document Distances

4.	

## TODO -- Project

1.	Due April 26th -- in class

2.	Project Report -- Hard Copy

3.	Demo to the TA -- will be scheduled b/w 4/26/2017 to 5/3/2017

### Report Formart

*	Technical Paper Style

*	Mention References

*	Write Logically

#### Content (5-6 Pages)

*	Title

*	Name

*	Abstract (100-200 words)

*	Introduction (problem description)

*	Related Work (by others, if any)

*	Approach (Justify it)

*	Implementation
	
	-	Data

	-	Software Architecture

*	Experiments and Results

*	Conclusions (Assessment of the complete project, your take on the whole project)

*	References



