"""
Test the trading Strategy Learner by selecting a stock to trade, period to trade
within, and iteratively averaging over the solution (c) 2017 Arjun Joshua
"""

import datetime as dt
import StrategyLearner as sl
import time

def test_code(verb = False):
    
    # instantiate the strategy learner
    learner = sl.StrategyLearner(verbose = verb)

    sym = "AAPL"
    repsOuter = 1 # number of times to re-create and re-train the QLearner
    sumCumReturn = 0 # initialize sum of all cumulative returns on the test
                     # data. This sum will be averaged over finally.
    start = time.time()
    for reps in range(repsOuter):
        print 'Outer rep = {}'.format(reps)
        # set parameters for training the learner
        stdate =dt.datetime(2008,1,1)
        enddate =dt.datetime(2009,12,31)

        # train the learner
        learner.addEvidence(symbol = sym, sd = stdate, ed = enddate, \
                                sv = 100000)

        # set parameters for testing
        stdate =dt.datetime(2010,1,1)
        enddate =dt.datetime(2011,12,31)

        # test the learner
        repsInner = 1 # number of times to calculate cumulative return on the 
                     # test data, after training the learner, in the step above
        for i in range(repsInner):
            cumReturn = learner.testPolicy(symbol = sym, sd = stdate, \
                                           ed = enddate, sv = 100000)
            sumCumReturn += cumReturn
        
    stop = time.time()
    print 'Average Test cumulative return = {}'.format(sumCumReturn / \
                                                    (repsOuter * repsInner))
    print 'Elapsed total time = {} s'.format(round(stop - start, 4))
   
if __name__=="__main__":
    test_code(verb = False) # set verbose here. If True, prints and plots a lot.
