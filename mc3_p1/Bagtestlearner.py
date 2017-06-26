import numpy as np
import BagLearner as bl
import RTLearner as rt
import time
import matplotlib.pyplot as plt

if __name__ == '__main__':
    data = np.genfromtxt('Data/Istanbul.csv', delimiter = ','); 
    # NEXT LINE SPECIFIC ONLY TO Istanbul.csv
    data = data[1:,1:];# eliminate 1st row and 1st column containing col-labels and dates 
    np.random.shuffle(data)# shuffle in-place
    split = int(0.6*data.shape[0])#60-40 break into train-test sets
    trainX = data[:split,:-1]
    trainY = data[:split,-1]#last column is labels
    testX = data[split:,:-1]
    testY = data[split:,-1]#last column is labels

    # create a BagLearner of RTlearners and train them
    start = time.time()
    learner = bl.BagLearner(learner = rt.RTLearner, kwargs = {"leaf_size":10}, \
                            bags = 20, boost = False, verbose = False) # create BagLearner
    learner.add_Evidence(trainX, trainY) # train BagLearner

    Y = learner.query(trainX)# get the predictions
    rmse = np.sqrt( ( (Y - trainY)**2 ).sum() / trainY.shape[0] )
    corr = np.corrcoef(Y, trainY)
    print("In sample results")
    print("RMSE: ", rmse)
    print("corr: ", corr[0,1])
    
    Y = learner.query(testX)# get the predictions
    rmse = np.sqrt( ( (Y - testY)**2 ).sum() / testY.shape[0] )
    corr = np.corrcoef(Y, testY)
    print("Out of sample results")
    print("RMSE: ", rmse)
    print("corr: ", corr[0,1])
    stop = time.time()
    print('time (s) = {}'.format(stop - start))
    
    # Plot RMSE for training dataset vs. number of bags each containing an RTLearner
    maxBagSize = 20
    errTrain = np.zeros(maxBagSize);
    errTest = np.zeros(maxBagSize);
    reps = 100 # number of configurations to average over for RMSE calculation
    start = time.time()
    for size in range(maxBagSize):
        for i in range(reps): # average by shuffling & selecting different train/test sets from data
            np.random.shuffle(data) # shuffle in-place
            learner = bl.BagLearner(learner = rt.RTLearner, kwargs = {"leaf_size":10}, \
                            bags = size + 1, boost = False, verbose = False) # create BagLearner
            learner.add_Evidence(trainX, trainY) # train on shuffled trainX, trainY
            # training sample testing
            Y = learner.query(trainX)# get the predictions
            errTrain[size] += np.sqrt( ( (Y - trainY)**2 ).sum() / trainY.shape[0] )
            # test sample testing
            Y = learner.query(testX)# get the predictions
            errTest[size] += np.sqrt( ( (Y - testY)**2 ).sum() / testY.shape[0] )
            
        
    stop = time.time()
    errTrain = errTrain / reps # average over number of repetitions
    errTest = errTest / reps
    plt.plot(range(1, maxBagSize+1), errTrain, label = 'training')
    plt.plot(range(1, maxBagSize+1), errTest, label = 'test')
    plt.xlabel('number of bags'); plt.ylabel('root mean squared error')
    plt.title('Bag Learner with leaf size = 10')
    lg = plt.legend(loc = 'best')
    lg.draw_frame(False) # removes box around legend
    print 'time (s) = {}'.format(stop - start)