"""
Solely uses technical indicators developed in indicators.py to trade.
This strategy does not use any labels (c) 2017 Arjun Joshua
"""
from indicators import defineData, getSma, getMom, getBb, getCross, standardize, \
                        createOrder
from marketsim import testcode_marketsim # used to run marketsim.py to return
                        # cumulative_return and a Series of portfolio values
import matplotlib.pyplot as plt

def plotVline(order):
    """
    @Summary: Plots vertical lines for buy and sell orders
    @param order: Series, consists of the values -2, -1, 0, 1, 2 denoting 
                  buying/selling of 200 or 400 shares or just holding
    @returns nothing
    """
    
    
    for date in order.index[order == 1]: # for dates corr. to buy 200 shares
        plt.axvline(date, color = 'g', linestyle = '--')
        
    for date in order.index[order == -1]: # for dates corr. to sell 200 shares
        plt.axvline(date, color = 'r', linestyle = '--')
    
    for date in order.index[order == 2]: # for dates corr. to buy 400 shares
        plt.axvline(date, color = 'g')
        
    for date in order.index[order == -2]: # for dates corr. to sell 400 shares
        plt.axvline(date, color = 'r')
    
def tradingStrategy(signal, holdTime = 21):
    """
    @Summary: Creates an order from a trading signal using 1 of 2 possible strategies
    @param signal: Series, consists of the values -1, 0, 1 denoting sell, hold
                   or buy
    @param holdTime: int, holding period after a transaction
    @returns order: Series, consists of the values -2, -1, 0, 1, 2 denoting 
                  selling/buying of 200 or 400 shares (depending on the 
                  strategy selected by commenting out below), or just holding
    """
    
    
    numDays = signal.shape[0]
    day = 0; 
    order = signal * 0 # initialize a Series of zeros with the same date indices as signal
    currOrder = 0 # current order status, -1 (short), 0 (no position) or 1 (long)
    while day < numDays:
########## +/- 200 shares per transaction with 0, 200, -200 allowed positions
########## order can take values of  0, 1, -1 corresponding to 0, +/-200 shares
        if (currOrder < 1) and (signal[day] == 1): # current order status is
                                    # not long and signal to buy is given
            order[day] = 1 # buy 200
            currOrder += order[day]
            day += holdTime # after buying wait for hold period
        elif (currOrder > -1) and (signal[day] == -1): # current order status
                                    # is not short and signal to sell is given
            order[day] = -1 # sell 200
            currOrder += order[day]
            day += holdTime # after selling wait for hold period
        else:
            day += 1 # END OF +/- 200 TRADING STRATEGY 1

######### +/- 200 or +/- 400 shares per transaction with 0, 200, -200 allowed positions
######### order can take values of  0, 1, -1, 2, -2 corresponding to 0, +/-200, +/-400 shares
#        if signal[day] != 0: # if signal is 1 or -1
#    # if currOrder=0, order=signal. If currOrder = 1 or -1, order is 0, 2 or -2
#            order[day] = signal[day] - currOrder
#            currOrder += order[day]
#            if order[day] == 0: # if no order executed, go to next day
#                day += 1
#            else: # if order = -2, -1, 1, 2
#                day += holdTime # hold time
#        else: # if signal = 0, go to next day
#            day += 1 # END OF +/- 200 or +/- 400 TRADING STRATEGY 2

        if day >= numDays: # if we reach the end of the trading period
                           # redeem all outstanding positions
            if currOrder == 1:
                order[-1] = -1
            elif currOrder == -1:
                order[-1] = 1
    

    return order
    
def getBbIndicator(data):
    """
    @Summary: trivial wrapper method for getBb
    @param data: Series, contains price with date indices
    
    @returns indicator: Series, contains Bollinger band indicator values with
                        date indices
    @returns bbWindow: int, window used in this method
    """
    
    
    bbWindow = 20 #10, 48, unoptimized values
    indicator = getBb(data, bbWindow) # indicator is a Series
    return indicator, bbWindow
    
def getCrossIndicator(data):
    """
    @Summary: trivial wrapper method for getCross
    @param data: Series, contains price with date indices
    @returns indicator: Series, contains crossover indicator values with
                        date indices
    @returns crossWindow: int, window used in this method
    """
    
    
    crossWindow = 25 #25, optimized value of 25 on manual trading strategy 1
    indicator = getCross(data, crossWindow) # indicator is a Series
    return indicator, crossWindow
    
def getMomIndicator(data):
    """
    @Summary: trivial wrapper method for getMom
    @param data: Series, contains price with date indices
    @returns indicator: Series, contains momentum indicator values with
                        date indices
    @returns momWindow: int, window used in this method
    """
    
    
    momWindow = 10 #3, optimized value of 10 on manual trading strategy 1
    indicator = getMom(data, momWindow) # indicator is a Series
    return indicator, momWindow
    
def getSmaIndicator(data):
    """
    @Summary:  wrapper method for getSma to return price/sma indicator
    @param data: Series, contains price with date indices
    @returns indicator: Series, contains price/sma indicator values with
                        date indices
    @returns smaWindow: int, window used in this method
    """
    
    
    smaWindow = 60# 60, optimized value of 60 on manual trading strategy 1
    sma = getSma(data, smaWindow)
    indicator = data / sma - 1 # indicator is a Series
    indicator.name = 'smaIndicator' # set column name
    return indicator, smaWindow
    
def testcode():
    
    
    data = defineData() # get AAPL between the in-sample dates set as default
    holdTime = 21 # in days
    
    smaIndicator, smaWindow = getSmaIndicator(data)
    smaThreshold = 0.012 #0.012 # optimized value on manual trading strategy 1
    # generate a buy signal (1) if price falls significantly below sma
    # generate a sell signal (-1) if prices rises significantly above sma
    smaSignal = 1 * (smaIndicator < -smaThreshold)  +  \
            -1 * (smaIndicator > smaThreshold)
    
    momIndicator, momWindow = getMomIndicator(data)
    momThreshold = 0.06 #0.055 # optimized value on manual trading strategy 1
    # generate a buy/sell signal if momentum is greatly positive/negative
    momSignal = -1 * (momIndicator < -momThreshold)  +  \
            1 * (momIndicator > momThreshold)
            
    bbWindow = 10#48 NOT OPTIMIZED
    bbIndicator = getBb(data, bbWindow)
    bbThreshold = 0#0.2 NOT OPTIMIZED
    # generate a buy/sell signal if indicator is below/above the lower/upper BB
    # and the indicator is rising/falling significantly
    bbSignal = -1 * ((bbIndicator > 1) & \
                     (standardize(data).diff(1) < -bbThreshold)) + \
                 1 * ((bbIndicator < -1) & \
                     (standardize(data).diff(1) > bbThreshold))
                 
    crossIndicator, crossWindow = getCrossIndicator(data)
    crossThreshold = 0.08 #0.08 # optimized value on manual trading strategy 1
    # generate a buy/sell signal if indicator is close to 0.5/-0.5
    crossSignal = 1 * ( (crossIndicator - 0.5).abs() < crossThreshold) + \
            -1 * ( (crossIndicator + 0.5).abs() < crossThreshold )

    # Combine individual signals. bbSignal is neglected here since including it
    # with the other signals did not result in label-free trading using strategy 1
    signal = 1 * ( (smaSignal == 1) & (momSignal ==1 ) & (crossSignal == 1) ) \
        + -1 * ( (smaSignal == -1) & (momSignal == -1) & (crossSignal == -1) )
    
    order = tradingStrategy(signal, holdTime)
    createOrder(order, 'rule_based')
    cumReturn, portVals = testcode_marketsim('rule_based', verbose = False)
    print('Cumulative return [%]: ', round(cumReturn * 100, 4) )

    plt.figure(figsize = (10,10))
    plt.subplot(311)
    plt.plot(data / data[0], label = 'benchmark', color = 'k')
    plt.plot(portVals / portVals[0], label = 'rule-based')
    plt.xticks(rotation=30)
    plotVline(order)
    plt.title('rule-based with sma + mom + crossover indicators')
    lg = plt.legend(loc = 'best')
    lg.draw_frame(False)
    plt.ylabel('normalized')

    plt.subplot(312)
    plt.plot(smaSignal/2, label = 'sma')
    plt.plot(momSignal/1.3,'.', label = 'mom')
    plt.plot(crossSignal/1.1,'.', label = 'crossover')
    plt.plot(signal, label = 'overall signal')
    plt.xticks(rotation=30)
    plt.ylabel('indicator signals [a.u.]')
    lg = plt.legend(loc = 'center right')
    lg.draw_frame(False)
    
    plt.subplot(313)
    plt.scatter(momIndicator[signal==0], crossIndicator[signal==0], \
                           color = 'k', label = 'hold')
    plt.scatter(momIndicator[signal==1], crossIndicator[signal==1], \
                           color = 'g', label = 'buy')
    plt.scatter(momIndicator[signal==-1], crossIndicator[signal==-1], \
                           color = 'r', label = 'sell')
    lg = plt.legend(loc = 'best')
    lg.draw_frame(True)
    plt.xlabel('Momentum Indicator')
    plt.ylabel('Crossover Indicator')
    
if __name__ == '__main__':
    testcode()
