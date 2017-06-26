Develop trading strategies using Technical Analysis and test them using previously developed market simulator. Use also previously developed Random Tree learner to train and test a learning trading algorithm.

The relevant files in this directory to be run for the mc3p3 project are:
1. indicators.py
2. rule_based.py
3. ML_based.py

All the other files were imported from other directories of this course that were
previously created. The RTclassLearner.py, BagClassLearner.py and KNNclassLearner.py
were modified for classification by converting the relevant np.mean() statements to
stats.mode() in the original RTLearner.py, BagLearner.py and KNNLearner.py files.
