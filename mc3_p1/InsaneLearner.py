"""
A super-baglearner containing 20 baglearners, each containing
20 linear regression learners  (c) 2017 Arjun Joshua
"""
import BagLearner as bl
import LinRegLearner as lrl
import numpy as np


class InsaneLearner(object):
    
    
    def __init__(self, verbose = False):
        
        learnerList = []
        num = 20 # number of baglearners
        self.verbose = verbose
        for i in range(num):
             # create each baglearner containing 20 linear regression learners
            learnerList.append( bl.BagLearner(lrl.LinRegLearner, kwargs = {}, \
                bags = 20, verbose = self.verbose) )
            
        self.learnerList = learnerList
        self.num = num
        
        
    def add_Evidence(self, trainX, trainY):
        """
        @summary: Add training data and train individual baglearners in InsaneLearner
        @param trainX: ndarray, X training data with examples in rows & features in columns
        @param trainY: 1Darray, Y training data
        
        Returns: nothing but trains the individual linear regression learners
        """        

        
        for learner in self.learnerList:
            learner.add_Evidence(trainX, trainY)
            
            
    def query(self, testX):
        """
        @summary: Add test data to query individual learners in BagLearner
        @param testX: ndarray, X test data with examples in rows & features in columns
        
        Returns pred: 1Darray, the predicted labels
        """        
        
        
        pred = np.empty( (testX.shape[0], self.num) ) # initialize pred, no. of 
        # rows = no. of test examples, no of columns = no. of individual bag learners
        for col in range(self.num):
            # predictions for each baglearner in rows of pred
            pred[:,col] = self.learnerList[col].query(testX)
            
        return pred.mean(axis = 1) # return (column) mean of all learners in 1D-array