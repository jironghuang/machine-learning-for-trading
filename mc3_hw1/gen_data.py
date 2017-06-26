"""
Generates data for Random Tree learner to outperform Linear Regression 
Learner and vice versa. (c) 2017 Arjun Joshua
"""

import numpy as np

# this function returns a dataset (X and Y) that will work
# better for linear regresstion than random decision trees
def best4LinReg(seed = 5):
    """
        @summary: Generate linear data best for Linear Regression
        @param seed: int, seeds random number generation
        
        Returns: ndarray, ndarray, X and Y data. Y data is 1-dimensional
    """


    np.random.seed(seed)
    numRows = np.random.randint(10, 1001) # randomly select rows in 10-1000 range
    numCols = np.random.randint(2, 1001) # randomly select cols in 2-1000 range

### The commented lines below generate random data but since each column
### vector of X contains unequally spaced values, adding up all columns
### results in an imperfect linear dependence of Y (sum of the col vectors)
### on the X column vectors
#    X =  np.random.normal(size=(numRows, numCols))
#    X.sort(axis = 0)

### This code generates equally spaced values in each X columns resulting
### in perfect linear dependence of Y on the X column vectors. The
### roundabout way the Xcol below are concatenated is because a 1-D array
### cannot be easily converted into an column vector in python. In particular
### operations like np.hstack / np.vstack / np.append / np.concatenate
### seem to assume that a 1-D array like Xcol is a row vector.
    Xcol = np.linspace(-2, 2, numRows) # can as well use any values other than -2, 2
    for row in range(numCols - 1):
        X = np.vstack((Xcol, Xcol))
        
    X = X.T
    
    Y = np.zeros( X.shape[0] )
    print 'shape of X = {}, shape of Y = {}'.format(np.shape(X), np.shape(Y))
    for col in range(X.shape[1]):
        Y = Y + X[:,col] # since linear dependence should be best for linear regression
    return X, Y

def best4RT(seed = 5):
    """
        @summary: Generate nonlinear (quadratic) data which should work better
                  for Random Decision Trees compared to Linear Regression
        @param seed: int, seeds random number generation
        
        Returns: ndarray, ndarray, X and Y data. Y data is 1-dimensional
    """


    np.random.seed(seed)
    numRows = np.random.randint(10, 1001)
    numCols = np.random.randint(2, 1001)
### See comments above for best4LinReg. The sort command is not used here
### because it seemed that sorting spoilt the nonlinear dependence defined
### below of Y on Xcol    
#    X =  np.random.normal(size=(numRows, numCols))
 
    Xcol = np.linspace(-2, 2, numRows)
    for row in range(numCols - 1):
        X = np.vstack((Xcol, Xcol))
        
    X = X.T
    
    Y = np.zeros( X.shape[0] )
    print 'shape of X = {}, shape of Y = {}'.format(np.shape(X), np.shape(Y))
    for col in range(X.shape[1]):
        Y = Y + X[:,col]**2 # since nonlinear dependence should be better for RT compared to Linear Regression

    return X, Y

if __name__=="__main__":
    print "gen_data.py is meant to be called by another file and not to be \
run independently!"
