"""
ML approach to trading using Random Decision Trees and Bagging (c) Arjun Joshua

"""
import pandas as pd
import numpy as np
from indicators import defineData, createOrder, standardize
from rule_based import getSmaIndicator, getMomIndicator, getCrossIndicator, \
                        getBbIndicator, plotVline, tradingStrategy
from marketsim import testcode_marketsim
import RTclassLearner as rt
import BagClassLearner as bl
import matplotlib.pyplot as plt
import time

def optimizeY(prices):
    """
    @Summary: Optimize Ybuy/Ysell parameters by plotting averaged error & 
              cumulative return for training/cross-val datasets vs. Ybuy 
              (assuming Ybuy = Ysell)
    @param prices: Series, contains adj. close price with date indices
    @returns nothing
    """
    
    
    numParam = 13
    tryParam = np.logspace(-4, -1, num = numParam) # values of Ybuy = Ysell
                                                    # to iterate over
    fname = 'ML_based' # name of orders file
    cumRetTrain = np.zeros(numParam)
    cumRetCrossVal = np.zeros(numParam)
    accTrain = np.zeros(numParam)
    accCrossVal = np.zeros(numParam)
    learner = rt.RTclassLearner(leaf_size = 5, verbose = False)# create learner
    reps = 100 # number of RTclassLearners whose predictions to average over
    start = time.time()
    for i in range(numParam): # X is a ndarray, Y is a 1darray
        X, Y, dates = getXY(prices, Ybuy = tryParam[i], Ysell = tryParam[i])
        split = int(0.6 * len(Y)) # divide into train and cross-val sets
        trainX = X[:split - 1, :]
        trainY = Y[:split - 1]
        trainDates = dates[:split - 1]
        crossValX = X[split:, :]
        crossValY = Y[split:]
        crossValDates = dates[split:]
        print tryParam[i] # print parameter values during computation
        for j in range(reps):
            learner.add_Evidence(trainX, trainY) # train on trainX, trainY
            # training sample querying
            pred = learner.query(trainX) # get the predictions
            accTrain[i] += np.sum(pred == trainY) / float( len(trainY) ) # accuracy
            signal = pd.Series(pred, index = trainDates) # create Series from predictions
            order = tradingStrategy(signal)
            createOrder(order, fname) # write order to disk
            cumReturn, portVals = testcode_marketsim(fname, verbose = False)
            cumRetTrain[i] += cumReturn
            # cross-validation sample querying
            pred = learner.query(crossValX) # get the predictions
            accCrossVal[i] += np.sum(pred == crossValY) / float( len(crossValY) )
            signal = pd.Series(pred, index = crossValDates)
            order = tradingStrategy(signal)
            createOrder(order, fname)
            cumReturn, portVals = testcode_marketsim(fname, verbose = False)
            cumRetCrossVal[i] += cumReturn
                
            
    stop = time.time()
    print 'Time to calculate cumulative return & error vs the selected parameters: {} s' \
                                                        .format(stop - start)
    plt.figure(figsize = (6,8))
    plt.subplot(211)
    plt.semilogx(tryParam * 100, cumRetTrain * 100 / reps, '.-', label = 'training')
    plt.semilogx(tryParam * 100, cumRetCrossVal * 100 / reps, '.-', label = 'cross-val')
    plt.ylabel('cumulative return [%]')
    plt.title('Random Decision Tree (leaf size = 5)')
    lg = plt.legend(loc = 'best')
    lg.draw_frame(False)
        
    plt.subplot(212)
    plt.semilogx(100 * tryParam, 100 - 100 * accTrain/reps, '.-', label = 'training')
    plt.semilogx(100 * tryParam, 100 - 100 * accCrossVal/reps, '.-', label = 'cross-val')
    plt.xlabel('Ybuy or Ysell [% change]'); plt.ylabel('error [%]')
    lg = plt.legend(loc = 'best')
    lg.draw_frame(False)
       
def optimizeParam(X, Y, dates):
    """
    @Summary: Optimize leaf size of RTclassLearner or number of bags of 
              BagClassLearner by plotting averaged error and cumulative return 
              for training & cross-val datasets vs. leaf size / number of bags 
    @param X: ndarray, of technical indicators. Examples in rows, features in 
              columns
    @param Y: 1darray, of the labels -1 (sell), 0 (hold), and 1 (buy) based on
              future returns after a holding time period
    @returns nothing
    """
    
    
    numParam = 1
    tryParam = np.linspace(75, 75, num = numParam) # values of leaf size / number
                                                  # of bags to iterate over
    fname = 'ML_based' # name of orders file
    cumRetTrain = np.zeros(numParam)
    cumRetCrossVal = np.zeros(numParam)
    accTrain = np.zeros(len(tryParam))
    accCrossVal = np.zeros(len(tryParam))
    reps = 1000 # number of configurations of RTclassLearners or 
                #BagClassLearners whose predictions to average over
    split = int(0.6 * len(Y)) # divide into train and cross-val sets
    trainX = X[:split - 1, :]
    trainY = Y[:split - 1]
    trainDates = dates[:split - 1]
    crossValX = X[split:, :]
    crossValY = Y[split:]
    crossValDates = dates[split:]
    start = time.time()
    for i in range(numParam): # initialize either RTclassLearner/BagClassLearner
        learner = rt.RTclassLearner(leaf_size = tryParam[i], verbose = False)# create learner
#        learner = bl.BagClassLearner(learner = rt.RTclassLearner, kwargs = {"leaf_size":75}, \
#                            bags = int(tryParam[i]), boost = False, verbose = False) # create BagClassLearner
        print tryParam[i] # print parameter values during computation
        for j in range(reps):
            learner.add_Evidence(trainX, trainY) # train on trainX, trainY
            # training sample querying
            pred = learner.query(trainX) # get the predictions
            accTrain[i] += np.sum(pred == trainY) / float( len(trainY) ) # accuracy
            signal = pd.Series(pred, index = trainDates) # create Series from predictions 
            order = tradingStrategy(signal)
            createOrder(order, fname) # write order to disk
            cumReturn, portVals = testcode_marketsim(fname, verbose = False)
            cumRetTrain[i] += cumReturn
            # cross-validation sample querying
            pred = learner.query(crossValX) # get the predictions
            accCrossVal[i] += np.sum(pred == crossValY) / float( len(crossValY) )
            signal = pd.Series(pred, index = crossValDates)
            order = tradingStrategy(signal)
            createOrder(order, fname)
            cumReturn, portVals = testcode_marketsim(fname, verbose = False)
            cumRetCrossVal[i] += cumReturn
            
        
    stop = time.time()
    print 'Time to calculate error vs the selected parameter: {} s'.format( \
                                                        round( stop - start, 2))
    plt.figure(figsize = (6,8))
    plt.subplot(211)
    plt.plot(tryParam, cumRetTrain * 100 / reps, '.-', label = 'training')
    plt.plot(tryParam, cumRetCrossVal * 100 / reps, '.-', label = 'cross-val')
    plt.ylabel('cumulative return [%]')
    plt.title('RT Learner (Ybuy = Ysell = 1 %)') # CHECK Ybuy/Ysell value here
    lg = plt.legend(loc = 'best')
    lg.draw_frame(False)
    print('Cumulative return [%]: ', round(cumRetCrossVal[-1] * 100 / reps, 4) )
    
    plt.subplot(212)
    plt.plot(tryParam, 100 - 100 * accTrain/reps, '.-', label = 'training')
    plt.plot(tryParam, 100 - 100 * accCrossVal/reps, '.-', label = 'cross-val')
    plt.xlabel('leaf size'); # CHECK leaf size OR number of bags
    plt.ylabel('error')
    lg = plt.legend(loc = 'best')
    lg.draw_frame(False)
    print ('Error [%]: ', round(100 - 100 * accCrossVal[-1] / reps, 4) )
    
def classify(price, Ybuy = 0.01, Ysell = 0.01, holdTime = 21):
    """
    @Summary: Label prices as buy (1), sell (-1), hold (0) based on future returns
    @param price: Series, contains adj. close price with date indices
    @param Ybuy/Ysell: float, threshold values of the ratio of future price to
                        the present price on the basis of which to classify 
                        the present price
    @param holdTime: int, used to classify present prices based on return after 
                    holdTime number of days
    @returns labels, Series of integers -1, 0 or 1
    """


    dates = price.index
    labels = pd.Series(np.zeros(len(dates)), index = dates)
    labels.name = 'labels'
    for dt in range(len(dates) - holdTime):
        if price[dt] / price[dt + holdTime] - 1 <= -Ybuy:
            labels.ix[dt] = 1
        elif price[dt] / price[dt + holdTime] - 1 >= Ysell:
            labels.ix[dt] = -1
    return labels
    
def getXY(prices, Ybuy = 0.01, Ysell = 0.01, holdTime = 21):
    """
    @Summary: Takes in a Series of prices and returns X features and Y labels
    @param prices: Series, contains adj. close price with date indices
    @param Ybuy/Ysell: float, threshold values of the ratio of future price to
                        the present price on the basis of which to classify 
                        the present price
    @param holdTime: int, used to classify present prices based on return after 
                    holdTime number of days
    @returns X: ndarray, of features based on standardized technical indicators
    @returns Y: 1darray, of labels -1, 0 and 1
    @returns dates
    """


    smaIndicator, smaWindow = getSmaIndicator(prices)
    momIndicator, momWindow = getMomIndicator(prices)
    crossIndicator, crossWindow = getCrossIndicator(prices)
    bbIndicator, bbWindow = getBbIndicator(prices)
    labels = classify(prices, Ybuy, Ysell, holdTime)
    df = pd.DataFrame(index = prices.index)
    # concatenate standardized technical indicators and corresponding labels
    df = df.join([(smaIndicator), (momIndicator), (crossIndicator), labels])
    df.dropna(inplace = True) # drop any rows having some NaN values
    X = df.values[:,0:-1]
    Y = df.values[:, -1]
    dates = df.index
    return X, Y, dates
    
def testcode():
    
    
    prices = defineData(startDate = '01-01-2008', stopDate = '31-12-2009', \
                           symList = ['AAPL']) # in-sample dates
#    optimizeY(prices) # UNCOMMENT to optimize Ybuy/Ysell
    Ybuy = 0.01; Ysell = 0.01; holdTime = 21
    X, Y, dates = getXY(prices, Ybuy, Ysell, holdTime) # X is an ndarray, Y is
                                                        # a 1darray
#    optimizeParam(X, Y, dates) # UNCOMMENT to optimize leaf size of RTclassLearner
                                # or numher of bags of BagClassLearner
                              
##### create a learner
    learner = rt.RTclassLearner(leaf_size = 75, verbose = False)# create learner
    split = int(0.6 * len(Y)) # divide into training and cross-val sets
    trainX = X[:split - 1, :]
    trainY = Y[:split - 1]
    trainDates = dates[:split - 1]
    crossValX = X[split:, :]
    crossValY = Y[split:]
    crossValDates = dates[split:]
   
#####Training on the entire in-sample data (USE WITH CAUTION)
#    learner.add_Evidence(X, Y)# train it
#    pred = learner.query(X)# get the predictions
#    accuracy = np.sum(pred == Y) / float( len(Y) )
#    print("In sample results")
#    print ('Error [%]: ', round(100 - 100 * accuracy, 4) )
#    # Generate order from in-sample predictions
#    signal = pd.Series(pred, index = dates)
#    order = tradingStrategy(signal)
#    createOrder(order, 'ML_based')
#    cumReturn, portVals = testcode_marketsim('ML_based', verbose = False)
#    
#    # Plot entire in-sample data, training indicators and their predictions (USE WITH CAUTION)
#    plt.figure(figsize = (11,13))
#    plt.subplot(411)
#    plt.plot(prices / prices[0], color = 'k', label = 'benchmark')
#    plt.plot(portVals / portVals[0], color = 'g', label = 'ML-based')
#    plt.xticks(rotation=30)
#    plt.ylabel('normalized')
#    plt.title('cumulative return = {} %'.format(round(cumReturn * 100)))
#    plotVline(order)
#    lg = plt.legend(loc = 'best')
#    lg.draw_frame(False)
#
#    plt.subplot(412)
#    plt.plot(signal / 2)
#    plt.xlim((prices.index[0], prices.index[-1]))
#    plt.xticks(rotation=30)
#    
#    plt.subplot(413)
#    plt.scatter(X[Y == 0, 1], X[Y == 0, 2], color = 'k', label = 'hold')
#    plt.scatter(X[Y == 1, 1], X[Y == 1, 2], color = 'g', label = 'buy')
#    plt.scatter(X[Y == -1, 1], X[Y == -1, 2], color = 'r', label = 'sell')
#    plt.title('in-sample data and labels before training')
#    lg = plt.legend(loc = 'best')
#    lg.draw_frame(True)
#    
#    plt.subplot(414)
#    plt.scatter(X[pred == 0, 1], X[pred == 0, 2], color = 'k', label = 'hold')
#    plt.scatter(X[pred == 1, 1], X[pred == 1, 2], color = 'g', label = 'buy')
#    plt.scatter(X[pred == -1, 1], X[pred == -1, 2], color = 'r', label = 'sell')
#    plt.xlabel('Momentum indicator'); plt.ylabel('Crossover indicator')
#    plt.title('in-sample data and predictions after training')
#    lg = plt.legend(loc = 'best')
#    lg.draw_frame(True)
#    
######out of sample testing after training on training set
    testPrices = defineData('01-01-2010', '31-12-2011') # out-of-sample dates
    testX, testY, testDates = getXY(testPrices, Ybuy, Ysell, holdTime) # testX
                                            # is an ndarray, testY is a 1darray
    cumRetTest = 0.0
    accTest = 0.0
    reps = 1 # number of configuratiokns of RTclassLearners whose predictions
            # to average over
    start = time.time()
    for j in range(reps):
        learner.add_Evidence(trainX, trainY)# train it
        pred = learner.query(testX)# get the predictions
        accTest += np.sum(pred == testY) / float( len(testY) ) # accuracy
        # Generate order from out of sample predictions
        signal = pd.Series(pred, index = testDates)
        order = tradingStrategy(signal)
        createOrder(order, 'ML_based')
        cumReturn, portVals = testcode_marketsim('ML_based', verbose = False)
        cumRetTest += cumReturn
        
    stop = time.time()
    print("Test sample results")
    print ('Error [%]: ', round(100 - 100 * accTest / reps, 4) )
    print('Cumulative return [%]: ', round(cumRetTest * 100 / reps, 4) )
    print('Elapsed time [s]: ', round(stop - start, 2))
    
    plt.plot(prices[:split - 1] / prices.values[0], label = 'train')
    plt.plot(prices[split:] / prices.values[0], label = 'cross-val')
    plt.plot(testPrices / prices.values[0], label = 'test')
    lg = plt.legend(loc = 'best')
    lg.draw_frame(False)
    plt.xticks(rotation=30)
    
        
if __name__ == '__main__':
    testcode()
    

