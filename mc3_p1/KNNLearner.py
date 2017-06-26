"""
An algorithm for k-nearest neighbours.  (c) 2017 Arjun Joshua
"""
import numpy as np

class KNNLearner(object):
    
    def __init__(self, k=3, verbose=False):
        self.k = k
    
    def add_evidence(self, dataX, dataY):
        """
        @summary: Add training data
        @param dataX: X training data
        @param dataY: Y training data
        """
        self.dataX = dataX
        self.dataY = dataY[:,None] #converts 1d to 2d array having 1 column
        
    def query(self, points):
        """
        @summary: Estimate a set of test points given the model we built.
        @param points: should be a numpy array with each row corresponding to a specific query.
        @returns the estimated values according to the saved model.
        """
        pred = np.zeros(points.shape[0]) #initialize prediction vector
        for i in range(0, points.shape[0]): #iterate over each test example
            sqDist = np.zeros(np.shape(self.dataY)) #initialize squared distances vector
            for j in range(0,self.dataX.shape[1]):
                sqDist[:,0] += (points[i,j] - self.dataX[:,j])**2
            
            sqDist = np.concatenate((sqDist, self.dataY), axis=1)
            sqDist = np.asarray(sorted(sqDist, key=lambda x:x[0]))
            pred[i] = np.mean(sqDist[0:self.k,1])
            
        return pred
            
if __name__ == 'main':
    print 'Not supposed to run this file. It is probably to be called by testlearner.py'
