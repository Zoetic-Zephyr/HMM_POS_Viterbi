The most standard breakdown for training and test purposes of the Penn
Treebank Corpus is:

Sections 02-21 Training
Section 23 Test
Section 24 Development

The other sections (00, 01, 22) are typically not used, although
section 00 has a training/development feel to it (many papers cite
examples from 00 files).

There are 2 possible versions of each file:

1) file.pos -- there are two columns separated by a tab:
   1st column: token
   2nd column: POS tag
   Blank lines separate sentences.

   This is the format of training files, system output, and development
   or test files used for scoring purposes.

2) file.words -- one token per line, with blank lines between sentences.
   Format of an input file for a tagging program.

For HW4, we are distributing the following files:

WSJ_02-21.pos  -- to use as the training corpus

WSJ_24.words   -- to use as your development set (for testing your system)

WSJ_24.pos     -- to use to check how well your system is doing

WSJ_23.words -- to run your system on.  You should produce a file in
	     	the .pos format as your output and submit it as per the
		submission instructions to be announced.

score.py -- this is a scorer which you can use on your development corpus. The scoring command is: 

python score.py WSJ_24.pos WSJ_24_sys.pos

assuming that your system output is called WSJ_24_sys.pos

This will give you an accuracy score. For further debugging and
tuning, I suggest using the UNIX diff utility.
