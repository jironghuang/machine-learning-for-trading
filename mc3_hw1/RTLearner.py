"""
An algorithm for a random tree learner.  (c) 2017 Arjun Joshua
"""
import numpy as np

class RTLearner(object):

    def __init__(self, leaf_size=1, verbose=False):
        self.leaf_size = leaf_size
        self.verbose = verbose


    def addEvidence(self, trainX, trainY):
        """
        @summary: Add training data and train random tree model
        @param trainX: ndarray, X training data with examples in rows & features in columns
        @param trainY: 1Darray, Y training data
        
        Returns: nothing but creates the Decision Tree matrix in dtMatrix
        """


        def build_tree(trainX, trainY): # cannot be called from outside this class
            """
            @summary: Builds the Decision Tree recursively by randomly choosing feature to split on and the splitting value
            @param trainX: ndarray, X training data at each node with examples in rows & features in columns
            @param trainY: 1Darray, Y training data at each node
        
            Returns dtMatrix: ndarray, the Decision Tree matrix
            """

            if trainX.shape[0] <= self.leaf_size: # if no. of train examples < leaf_size, aggregate into leaf
                return np.array([[-1, trainY.mean(), np.nan, np.nan]])
            elif (trainY[:]==trainY[0]).all(): # if training labels identical, return leaf
                return np.array([[-1, trainY[0], np.nan, np.nan]])# return leaf
            else:
                factors = range(trainX.shape[1]) # returns list of integers
                i = 0
                while i < 5:
                    i += 1
                    feat = np.random.choice(factors) # randomly choose feature to split on
                    splitVal = np.mean(np.random.choice(trainX[:,feat], 2, \
                             replace=False)) # randomly choose split value of feature
                    indLeft = trainX[:,feat]<=splitVal # logical array for indexing
                    indRight = ~indLeft # complement of indLeft
                    if (indLeft[:]==1).all() != True: # break out of while loop...
                        break #...if selected feature vector is divided into two groups
                        
                if i < 5 or (indLeft[:] == 1).all() != True: # if labels divided into 2 groups
                    leftree = build_tree(trainX[indLeft],trainY[indLeft])
                    rightree = build_tree(trainX[indRight], trainY[indRight])
                    root = np.array([[feat, splitVal, 1, leftree.shape[0]+1]])
                    dtMatrix = np.append(root, leftree, axis=0)
                    dtMatrix = np.append(dtMatrix, rightree, axis=0)
                else: # labels cannot be divided into two groups, try again
                    if self.verbose == True:
                        print('Node appears to be terminal with feature = {}, \
                              splitVal = {}: 5 attempts made yet cannot partition node, \
                                retrying...!'.format(feat, splitVal))

                    dtMatrix = build_tree(trainX[indLeft],trainY[indLeft])

            return dtMatrix

            
        dtMatrix = build_tree(trainX, trainY) # Decision Tree Matrix
        self.dtMatrix = dtMatrix
        if self.verbose == True:
            print dtMatrix
            print np.shape(dtMatrix)


    def query(self, testX):
        """
        @summary: Add test data to recursively query decision tree model
        @param testX: ndarray, X test data with examples in rows & features in columns
        
        Returns pred: 1Darray, the predicted labels
        """

        def binary_search(line):
            """
            @summary: Searches the decision tree matrix
            @param line: int, the row of the decision tree matrix to search
        
            Returns label: 1Darray, the predicted labels
            """

            feat, splitVal = self.dtMatrix[line,0:2] # read feature that was split on & the corresponding splitting value
            if feat == -1:
                return splitVal# return leaf
            elif testX[row, int(feat)] <= splitVal:
                label = binary_search(int( self.dtMatrix[line,2] ) + line) # search left tree
            else:
                label = binary_search(int( self.dtMatrix[line,3] ) + line)# search right tree
            
            return label

        pred = np.empty(testX.shape[0]) # initialize pred
        for row in range(0, np.shape(testX)[0]): # for each test example (row)
            pred[row] = binary_search(line=0) # start recursive search at 1st line of the decision tree matrix dtMatrix

        return pred