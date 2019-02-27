******************************************************************************************
*                                                                                        *
*            Hidden Markov Model Part of Speech Tagging using Viterbi Algorithm          *
*                             HMM POS Tagging using Viterbi                              *
*                                                                                        *
*                                       Created By:                                      *
*                         Zheng(Jack) Zhang & Fangqing(Quinn) He                         *
*                                        2019/2/26                                       *
*                                                                                        *
******************************************************************************************

====================================
How to Run the system?

It's easy!
All you have to do is running the Python script named: zz1444_fh805_viterbi_HW3.py
You will see the Part of Speech Tagged output based on a Hidden Markov Model using Viterbi Algorithm in the folder named "output".

====================================
How does the system work?

Training Stage:
Two separate Python dictionaries, named "likelihood" and "transition", are built to store the likelihood and transition probability of words in the training corpus.
Each key in the likelihood dictionary is a POS tag, and the value is an internal dictionary that contains the words appeared under this tag and their respective appearance probability versus the total number of words under this tag. 
Each key in the transition dictionary is a POS tag, and the value is an internal dictionary that contains the POS tag that follows the key tag immediately and their respective appearance probability versus the total number of trailing tags under this tag.

A supplementary set named "words" and dictionary named "UNKNOWN_WORD" are constructed as well under this stage:
The set contains all the appeared word in the training corpus.
The keys in the dictionary are the POS tags of words that only appeared once in the training corpus and the values are their respective appearance probability versus the total number of appearances of this tag.
They set and dictionary above are used for handling OOV (out-of-vocabulary) words in the test file. The mechanisms behind will be explained in detail in the OOV handler section.

Tagging Stage:
1. Iterate over the lines in test file.
2. Treat the word separately if the word is the first word, OOV or the end of sentence in terms of likelihood probability and previous word Viterbi probability.
3. Get the product of the current word's likelihood probability, transition probability and its previous word Viterbi probability under different tags.
4. Record the maximum Viterbi probability under each tag, and respective previous word tag
5. Create a two-dimensional dictionary named "field", the key is the index of current word within the sentence and the value is the internal dictionary that record the valid tag and its maximum Viterbi probability.
6. Create a two-dimensional dictionary named "back_pointer", the key is the index of current word within the sentence and the value is the internal dictionary that record the valid ta and its respective backpointer.
7. For each sentence, determine the tag according to current tag and its backpointer.

====================================
How we handled OOV?

OOV (out-of-vocabulary) words have to be carefully handled, otherwise they will crash the entire system for the intuitive reason that there is no key in the both the "likelihood" and "transition" dictionary for such words.

To deal with this problem, we first implemented a simple/naive version of OOV handler by manually setting all the Viterbi probability of words that do not appear in the training corpus to an arbitrary value, e.g. 1/1000. This can work, but the POS tagging accuracy achieved using this method can be improved.

====================================
What did we do as extra work?

1. To improve the accuracy, we also implemented a better version of OOV handler. First, find out all the words that only appear once in the training corpus. Then extract their tags as the keys in the "UNKNOWN_WORD" dictionary and their respective value are the appearance probability versus the total number of appearances of their respective tags. In this way, we are essentially treating all the OOV words as if they are a single word, and based on its most likely tag, we provide the likelihood probability. This increases tagging accuracy.

2. In addition, when testing our system using arbitrary documents, we noticed that sometimes the system will crash for unknown reason. Through intense hardcore debugging, we eventually find out that it is because, sometimes, when calculating the Viterbi probabilities for words, they can be so small (-10^231) that Python essentially think them they are ZERO. There are many different ways to deal with this situation, but a simple workaround is: to increase the probability by 10 times, and restore it again after comparison. With some research, we later find out that this is actually a common error in the field of Machine Learning. So we are happy to learn something new :)

3. We also handle the situation that the given word's viterbi probability stays zero no matter what tags combination we offer to the word. It turns out to be the issue of insufficient transition paths offered by training corpus. No previous tag can be followed by any tag current word under accrording to the "transition" dictionary. To solve this, we record the current tag with maximum likelihood and the previous tag with maximum viterbi probability. In this way, we manually set the "field" dictionary and the "backpointer" dictionary for this zero path word.

====================================

Work division?

Basically, we did everything together. But for sake of clarity, see below:

– Person 1&2: create the tables with the probabilities
– Person 1&2 create the initial version of Viterbi and a very simple OOV strategy (assume all POS have equal probability)
– Person 1&2: OOV strategy based on words occurring once
– Person 1&2: error analysis on development corpus to determine next improvements
– Person 1&2: Manual Rule based system
– Person 1&2: everything else