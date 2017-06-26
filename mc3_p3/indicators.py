"""
Development of a few standard technical indicators including a custom "crossover" 
indicator. Also contains a routine to evaluate the maximum possible return with 
a given stock looking into the future and with a restriction of +/- 200 shares 
per transaction with 0, 200, -200 being the only allowed positions
(c) 2017 Arjun Joshua
"""
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from marketsim import testcode_marketsim
from util import get_data

def fname_to_path(fname, base_dir = "./orders"):
    """
    @summary: Takes a filename, appends a .csv extenstion and prefixes a 
              base directory
    @param fname: string, the filename
    @param base_dir: string, the base directory
    @returns the composite string containing the filename and the base dir
    """
    
    
    return os.path.join(base_dir, "{}.csv".format(fname))

def createOrder(order, fname, base_dir = './orders', symbol = 'AAPL', \
                                                        unit = 200):
    """
    @summary: Takes an order, a filename, and a base directory and writes a csv
              order file
    @param order: Series, consists of the values -2, -1, 0, 1, 2 denoting 
                  buying/selling of 200 or 400 shares or just holding
    @param fname: string, the filename
    @param base_dir: string, the base directory
    @param symbol: string, the stock to trade
    @param unit: integer, number of stocks to trade per unit in the order Series
    @returns nothing but writes the file on disk
    """
    
    
    f = open(fname_to_path(fname, base_dir), 'w')
    f.write('Date,Symbol,Order,Shares\n')
    for ind, val in order.iteritems(): # iterator over the timestamp indices 
                                       # and values in 'order'
        if val == 1:
            f.write('{},{},BUY,{}\n'.format(ind.date(), symbol, unit))
        elif val == -1:
            f.write('{},{},SELL,{}\n'.format(ind.date(), symbol, unit))
        elif val == 2:
            f.write('{},{},BUY,{}\n'.format(ind.date(), symbol, 2 * unit))
        elif val == -2:
            f.write('{},{},SELL,{}\n'.format(ind.date(), symbol, 2 * unit))
    
    f.close
    
def bestPossibleStrategy(data):
    """
    @Summmary: Evaluate the maximum possible return with a given stock looking 
               into the future and with a restriction of +/- 200 shares per 
               transaction with 0, 200 (represented by 1), -200 (represented 
               by -1) being the only allowed positions (hold, buy 200 and sell 200)
    @param data: Series, contains adj. close prices of the stock with date indices
    @returns order: Series, consists of the values -1, 0, 1, denoting 
                  selling/buying of 200 shares or just holding
    """
    
    
    nextDayReturn = data.ix[:-1] / data.values[1:] - 1 # calculate today's price
     # relative to tomorrow's price. This is to decide whether to buy/sell today
    nextDayReturn = nextDayReturn.append(data[-1:]) # restore the last date/value
                                   # row which was removed by the previous step
    nextDayReturn[-1] = np.nan # The value of the last row is a NaN since we do
                               # not know the next day's price
    dailyOrder = -1 * nextDayReturn.apply(np.sign) # find the sign and invert it
     # In dailyOrder, 1 means buy today, and -1 means sell today
    order = dailyOrder.diff(periods = 1) / 2 # pick out only where 1 changes to
     # -1 and vice-versa while eliminating consecutive 1s and -1s. Division by
     # 2 needed since we are constrained to buying and selling only 200 shares at a time.
    order[0] = dailyOrder[0] # restore the first date/value row removed by the
     # previous differentiation operation

    # on the last day, close any open positions
    if order.sum() == -1:
        order[-1] = 1
    elif order.sum() == 1:
        order[-1] = -1
            
    return order

def standardize(data):
    """
    @Summary: Normalize by substracting mean and ividing by standard deviation
    @param data: DataFrame, Series, or ndarray
    @returns standardized data
    """
    
    
    mu = np.mean(data)
    sigma = np.std(data)
    return (data - mu) / sigma

def getCross(data, crossWindow = 25):
    """
    @Summary: My idea was to trigger buy/sell signals at the minima/maxima of
              a slowly oscillating time-series where it equals and crosses 
              its moving average but with an opposite sign of slope
    @param data: Series, contains prices with date indices
    @crossWindow: int, look-back window for simple moving average
    @returns Series with values ~ 0.5/-0.5 (buy/sell) on days where the 
            crossover occurs at the valleys/peaks of the  slow oscillations. On
            days where the slopes of the time-series and its sma have the same
            sign, the value is set to 0. On days where the slopes have opposite
            sign but a crossover does not occur, values between -1 and 1 are 
            returned
    """
    
    
    value = 1 / (1 + data / getSma(data, crossWindow)) # ranges from 0 to 1,
                                                    # usually around 0.5
    # set sign = 1 (buy) for prices going up in time and sma going down.
    # set sign = -1 (sell) for the opposite situation.
    sign = ( 1 * ( (data.diff(periods = crossWindow) > 0) & \
    (getSma(data, crossWindow).diff(periods = crossWindow) < 0) ) | \
                   -1 * ( (data.diff(periods = crossWindow) < 0) & \
    (getSma(data, crossWindow).diff(periods = crossWindow) > 0) ) )
    indicator = sign * value
    indicator.name = 'crossIndicator' # set column name
    return indicator

def getMom(data, momWindow = 5):
    """
    @Summary: Calculate momentum indicator = ratio of price with respect 
              to price momWindow number of days back - 1
    @param data: Series, contains prices with date indices
    @param momWindow: int, number of days to look back
    @returns Series of the same size as input data
    """
    
    
    diff = data.diff(periods = momWindow) # difference of prices wrt price
                                          # momWindow number of days back
    # divide above difference by price momWindow number of days back
    diff.ix[momWindow:] = diff.ix[momWindow:] / data.values[:-momWindow]
    diff.name = 'momIndicator' # set column name
    return diff

def getSma(data, smaWindow = 20):
    """
    @Summary: simple moving average
    @data: Series, contains price with date indices
    @smaWindow: int, number of days to look back to calculate the moving average
    @returns Series of the same size as input data
    """
    

    return data.rolling(window = smaWindow).mean()

def getBb(data, bbWindow = 20):
    """
    @Summary: Calculate Bollinger band indicator
    @param data: Series, contains price with date indices
    @param bbWindow: int, number of days to look back to calculate the moving
                     average and moving standard deviation
    @returns Series of the same size as input data
    """

    
    sma = getSma(data, bbWindow)
    indicator = (data - sma) / ( 2 * (data.rolling(window = bbWindow).std()) )
    indicator.name = 'bbIndicator' # set column name
    return indicator
    
def defineData(startDate = '01-01-2008', stopDate = '31-12-2009', symList = ['AAPL']):
    """
    @Summary: Create a Series of a single stock price
    @param startDate: starting date
    @param stopDate: end date
    @param symList: List of a single stock symbol
    @returns a Series containing the prices with the specified dates as indices
    """
    
    
    dates = pd.date_range(startDate, stopDate)
    df = get_data(symList, dates)
    data = df.ix[:,1] # First column is SPY by default

    return data
    
def testcode():
    
    
    data = defineData() # get AAPL between the in-sample dates set as default
    smaWindow = 60
    sma = getSma(data, smaWindow)
    bbWindow = 20
    bb = getBb(data, bbWindow)
    bbUpper = sma + 2 * data.rolling(window = bbWindow).std()
    bbLower = sma - 2 * data.rolling(window = bbWindow).std()
    momWindow = 10
    mom = getMom(data, momWindow)
    crossWindow = 25
    crossover = getCross(data, crossWindow)
        
    plt.figure(figsize = (12,14))
    plt.subplot(421)
    data.plot(label='adj. close')
    sma.plot(label='sma')
    plt.ylabel('prices')
    lg = plt.legend(loc = 'best')
    lg.draw_frame(False)
    
    plt.subplot(422)
    plt.plot(data / sma - 1)
    plt.ylabel('sma indicator')
    plt.xlim((data.index[0], data.index[-1]))
    plt.xticks(rotation=30)
    
    plt.subplot(423)
    data.plot(label='adj. close')
    plt.plot(bbUpper, label='upper BB')
    plt.plot(bbLower, label='lower BB')
    plt.ylabel('prices')
    lg = plt.legend(loc = 'best')
    lg.draw_frame(False)
    
    plt.subplot(424)
    plt.plot(bb)
    plt.ylabel('Bollinger band indicator')
    plt.xlim((data.index[0], data.index[-1]))
    plt.xticks(rotation=30)
    
    plt.subplot(426)
    plt.plot(mom)
    plt.ylabel('momentum')
    plt.xlim((data.index[0], data.index[-1]))
    plt.xticks(rotation=30)
   
    plt.subplot(427)
    plt.plot(1 / (1 + data / getSma(data, crossWindow)))
    plt.ylabel('crossover value')
    plt.xlim((data.index[0], data.index[-1]))
    plt.xticks(rotation=30)
    
    plt.subplot(428)
    plt.plot(crossover)
    plt.ylabel('crossover indicator')
    plt.xlim((data.index[0], data.index[-1]))
    plt.xticks(rotation=30)
    
    order = bestPossibleStrategy(data)
    createOrder(order, 'bestPossibleStrategy')
    cumReturn, portVals = testcode_marketsim('bestPossibleStrategy', verbose = False)
    print 'cumulative return of best possible strategy [%] = {}'.format(cumReturn * 100)
        
if __name__ == '__main__':
    testcode()
