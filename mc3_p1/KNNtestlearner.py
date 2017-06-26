"""
Test a learner.  (c) 2017 Arjun Joshua
"""

import numpy as np
import math
import KNNLearner as knn
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

if __name__=="__main__":
    # Replaced with different method
#    data = np.genfromtxt('Data/best4lrr_data.csv', delimiter=',')
    dataTrain = np.genfromtxt('mc3p1_data_spr2016/ripple.csv', delimiter=',')
    dataTest =  np.genfromtxt('mc3p1_data_spr2016/testcase10.csv', delimiter=',')


    # compute how much of the data is training and testing
#    train_rows = int(0.60* data.shape[0]) #math.floor(0.60* data.shape[0]) AJ commented out 'floor'
#    test_rows = data.shape[0] - train_rows

    # separate out training and testing data
#    trainX = data[:train_rows,0:-1]
#    trainY = data[:train_rows,-1]
#    testX = data[train_rows:,0:-1]
#    testY = data[train_rows:,-1]
    trainX = dataTrain[:,0:-1]
    trainY = dataTrain[:,-1]
    testX = dataTest[:,0:-1]
    testY = dataTest[:,-1]

#    print(testX.shape)
#    print(testY.shape)

    # create a learner and train it
    start = time.time()
    learner = knn.KNNLearner(k = 3, verbose = True) # create a KNNLearner
    learner.add_evidence(trainX, trainY) # train it

    # evaluate in sample
    Y = learner.query(trainX) # get the predictions
    rmse = math.sqrt(((trainY - Y) ** 2).sum()/trainY.shape[0])
#    print(learner.model_coefs) 
    print("In sample results")
    print("RMSE: ", rmse)
    corr = np.corrcoef(Y, y=trainY)
    print("corr: ", corr[0,1])
#
    # evaluate out of sample
    Y = learner.query(testX) # get the predictions
    rmse = math.sqrt(((testY - Y) ** 2).sum()/testY.shape[0])
    print
    print("Out of sample results")
    print("RMSE: ", rmse)
    corr = np.corrcoef(Y, y=testY)
    print("corr: ", corr[0,1])
    stop = time.time()
    print 'time = {} s'.format(stop-start)
    
    print 'number of dependent variables = {}'.format(np.shape(trainX)[1])
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(trainX[:,0], trainX[:,1], trainY, c='c')
    ax.scatter(testX[:,0], testX[:,1], testY, c='r')

#    plt.subplot(121)
#    plt.scatter(trainX[:,0], trainY)
#    plt.scatter(testX[:,0], testY, c='r')
#    plt.subplot(122)
#    plt.scatter(trainX[:,1], trainY)
#    plt.scatter(testX[:,1], testY, c='r')