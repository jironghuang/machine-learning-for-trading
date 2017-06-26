"""
An algorithm for a bootstrap aggregating learner.  (c) 2017 Arjun Joshua
"""
import numpy as np
from scipy import stats


class BagClassLearner(object):
    
    
    def __init__(self, learner, kwargs, bags = 1, boost = False, verbose = False):

        self.verbose = verbose
        if self.verbose == True:
            for name, value in kwargs.items(): # print name-value pairs contained in kwargs dictionary
                print '{0} = {1}'.format(name, value)
                
        learnerList = [] # initialize list of learner objects
        for bag in range(bags):
            # unpack kwargs dictionary to create a learner object
            learnerList.append(learner(verbose = self.verbose, **kwargs))
            
        self.learnerList = learnerList
        self.bags = bags
        self.boost = boost
        
        
    def add_Evidence(self, trainX, trainY):
        """
        @summary: Add training data and train individual learners in BagLearner
        @param trainX: ndarray, X training data with examples in rows & features in columns
        @param trainY: 1Darray, Y training data
        
        Returns: nothing but possibly trains the individual learners depending 
                 on what add_Evidence does to each one
        """        
        
        
        bagSize = len(trainY) # bag size determined by number of training examples
        for learner in self.learnerList:
            # randomly select indexes of training examples. Number of indexes
            # selected determined by bagSize
            ix = np.random.choice( range( bagSize ), bagSize, replace = True )
            bagX = trainX[ix]; bagY = trainY[ix] # training examples/labels for each BagLearner
            learner.add_Evidence(bagX, bagY) # add training examples/labels to each BagLearner
        
        
    def query(self, testX):
        """
        @summary: Add test data to query individual learners in BagLearner
        @param testX: ndarray, X test data with examples in rows & features in columns
        
        Returns pred: 1Darray, the predicted labels
        """

        
        pred = np.empty((testX.shape[0],self.bags)) # initialize pred, no. of 
        # rows = no. of test examples, no of columns = no. of individual learners
        for col in range(pred.shape[1]):
            # predictions for each learner in rows of pred
            pred[:,col] = self.learnerList[col].query(testX)
            
        modeValue, binCount = stats.mode(pred, axis = 1) # mode and number of 
        # counts along columns (i.e. over all learners) returned as column vectors
        return  modeValue[:,0] # return (column) mode of all learners in 1D-array
        