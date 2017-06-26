import numpy as np
import InsaneLearner as it

if __name__ == '__main__':
    data = np.genfromtxt('Data/best4lrr_data.csv', delimiter = ',')
#    np.random.shuffle(data) # Don't shuffle to compare output with single linreglearner
    split = int( 0.6 * data.shape[0] ) # 60-40 break into train-test sets
    trainX = data[:split, :-1]
    trainY = data[:split, -1] # last column is labels
    testX = data[split:, :-1]
    testY = data[split:, -1] # last column is labels
    
    learner = it.InsaneLearner(verbose = False) # constructor for InsaneLearner
    learner.add_Evidence(trainX, trainY)
    
    Y = learner.query(trainX) # get the predictions
    rmse = np.sqrt( ( (Y - trainY)**2 ).sum() / trainY.shape[0] )
    corr = np.corrcoef(Y, trainY)
    print("In sample results")
    print("RMSE: ", rmse)
    print("corr: ", corr[0,1])
    
    Y = learner.query(testX) # get the predictions
    rmse = np.sqrt( ( (Y - testY)**2 ).sum() / testY.shape[0] )
    corr = np.corrcoef(Y, testY)
    print
    print("Out of sample results")
    print("RMSE: ", rmse)
    print("corr: ", corr[0,1])