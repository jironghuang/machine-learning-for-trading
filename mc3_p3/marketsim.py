"""MC2-P1: Market simulator."""

import pandas as pd
import numpy as np
from util import get_data, symbol_to_path

import sys
sys.path.insert(0, '../mc1_p1')
from analysis import get_portfolio_stats, plot_normalized_data

def get_leverage(date, df, prices):
    leverage = np.sum( abs( prices.ix[date, 1:].multiply( df.ix[date, 1:] ))) / \
( np.sum( prices.ix[date, 1:].multiply( df.ix[date, 1:] )) + df.ix[date, 'cash'] )
    
    return leverage


def get_position_values(cashStocks, prices, symbols):
    posVals = cashStocks.copy()
    posVals.ix[:, 1:] = cashStocks.ix[:, 1:] * prices.ix[:, 1:]
    return posVals
    

def get_cash_stocks(order, prices, symbols, start_val, leverLimit = True):
    df = pd.DataFrame(0, index = prices.index, columns = ['cash'] + symbols)
    df.ix[:,0] = start_val
    
    for row in order.itertuples():
        date, sym, tradingSignal, n = row #extract date, symbol,signal, and no. of shares
        sharePrice = prices.ix[date, sym]
        if tradingSignal == 'SELL':
            n = -n
        
        df.ix[date:, sym] += n
        df.ix[date:, 'cash'] += - n * sharePrice
        
        leverage = get_leverage(date, df, prices)
        if (leverage > 3.0) & leverLimit: #reverse entries made above in df
            df.ix[date:, sym] += - n
            df.ix[date:, 'cash'] += n * sharePrice
            print '\nLeverage of 3 exceeded. See line in order file: {}-{}-{},{},{},{}'.format( \
                    date.year, date.month, date.day, sym, tradingSignal, abs(n))
            
            
    return df

    
def compute_portvals(orders_file = "./orders/orders2.csv", \
                     start_val = 1000000, leverLimit = True):
    # this is the function the autograder will call to test your code
    order = pd.read_csv(orders_file, index_col = 'Date', parse_dates = True, 
                          na_values=['nan'])
    
    """Create dataframe prices with symbol/dates/prices relevant to the order"""
    start_date = order.index[0]
    end_date = order.index[-1]
    dates = pd.date_range(start_date, end_date)
    symbols = list( order.ix[:,0].unique() ) #ndarray to list of symbols in order
    prices = get_data(symbols, dates)
    
    """Create dataframe of cash and deposits in stocks, indexed by date"""
    cashStocks = get_cash_stocks(order, prices, symbols, start_val, leverLimit)
    posVals = get_position_values(cashStocks, prices, symbols)
    portVals = posVals.sum(axis = 1)
    
    return portVals

def testcode_marketsim(symbol = 'ML_based', base_dir = './orders/', \
                       sv = 100000, leverLimit = True, verbose = True):
###    Use one of the order folders below    ###
#    of = "./orders/benchmark.csv"
#    of = "./orders/bestPossibleStrategy.csv"
#    of = "./orders/rule_based.csv"
#    of = "./orders/ML_based.csv"
    of = symbol_to_path(symbol, base_dir)

#    sv = 100000 # starting value of portfolio, i.e. initial cash available

    # Process orders
    portVals = compute_portvals(of, sv, leverLimit)
    if isinstance(portVals, pd.DataFrame):
        portVals = portVals[portVals.columns[0]] # just get the first column as a Series
    else:
        "warning, code did not return a DataFrame"
    
    start_date = portVals.index[0]
    end_date = portVals.index[-1]
    pricesSPX = get_data(['$SPX'], pd.date_range(start_date, end_date))
    pricesSPX = pricesSPX['$SPX']

    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = get_portfolio_stats(portVals, \
                                                daily_rf = 0, samples_per_year = 252)
    cum_ret_SPY, avg_daily_ret_SPY, std_daily_ret_SPY, sharpe_ratio_SPY = \
            get_portfolio_stats(pricesSPX, daily_rf = 0, samples_per_year = 252)

    # Compare portfolio against $SPX
    if verbose == True:
        dfTemp = pd.concat([portVals, pricesSPX], axis = 1, keys = ['portfolio', '$SPX'])
        plot_normalized_data(dfTemp,'', '', '')
    
        print "\nDate Range: {} to {}".format(start_date.date(), end_date.date())
        print
        print "Sharpe Ratio of Fund: {}".format(sharpe_ratio)
        print "Sharpe Ratio of SPY : {}".format(sharpe_ratio_SPY)
        print
        print "Cumulative Return of Fund: {}".format(cum_ret)
        print "Cumulative Return of SPY : {}".format(cum_ret_SPY)
        print
        print "Standard Deviation of Fund: {}".format(std_daily_ret)
        print "Standard Deviation of SPY : {}".format(std_daily_ret_SPY)
        print
        print "Average Daily Return of Fund: {}".format(avg_daily_ret)
        print "Average Daily Return of SPY : {}".format(avg_daily_ret_SPY)
        print
        print "Final Portfolio Value: {}".format(portVals[-1])
    
    return cum_ret, portVals        
if __name__ == "__main__":
    testcode_marketsim()
