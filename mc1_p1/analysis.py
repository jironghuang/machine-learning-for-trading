"""MC1-P1: Analyze a portfolio."""

import pandas as pd
import numpy as np
import datetime as dt
from util import get_data, plot_data

def get_portfolio_value(prices, allocs, start_val):
    """Return a single column of daily portfolio values given prices of stocks
    (no datetime object), allocations into each stock and starting value of 
    capital. Returns 'portVal' as a Series"""
    normed = prices/prices.values[0,:]
    alloced = allocs * normed
    posVal = start_val * alloced
    portVal = posVal.sum(1) # portVal is a Series
    return portVal

    
def get_portfolio_stats(portVal, daily_rf, samples_per_year):
    """Return Cumulative return, mean daily return, standard deviation of daily
    returns, and Sharpe ratio given portfolio values, daily risk-free rate, and
    number of samples per year"""
    cr = portVal[-1]/portVal[0]  - 1 # Cumulative return
    dr = portVal[1:]/portVal.values[0:-1] - 1 #dr is a Series of daily returns
    adr = dr.mean() # average daily return
    sddr = dr.std() # standard deviation of daily return
    sr = np.sqrt(samples_per_year) * ( adr - daily_rf ) / sddr # Sharpe ratio
    return cr, adr, sddr, sr
    
    
def plot_normalized_data(df, title, xlabel, ylabel):
    """Plots portfolio values, normalized to t=0, over time and compares vs. SPY.
    Uses the plot_data() utility function"""
    dfNorm = df / df.values[0,:]
    plot_data(dfNorm, title, xlabel, ylabel)
    
    
def assess_portfolio(sd = dt.datetime(2008,1,1), ed = dt.datetime(2009,1,1), \
    syms = ['GOOG','AAPL','GLD','XOM'], \
    allocs=[0.1,0.2,0.3,0.4], \
    sv=1000000, rfr=0.0, sf=252.0, \
    gen_plot=False):

    # Read in adjusted closing prices for given symbols, date range
    dates = pd.date_range(sd, ed)
    prices_all = get_data(syms, dates)  # automatically adds SPY
    prices = prices_all[syms]  # only portfolio symbols
    prices_SPY = prices_all['SPY']  # only SPY, for comparison later
    
    portVal = get_portfolio_value(prices, allocs, sv)
    cr, adr, sddr, sr = get_portfolio_stats(portVal, rfr, sf)

    # Compare daily portfolio value with SPY using a normalized plot
    if gen_plot:
        # add code to plot here
        df_temp = pd.concat([portVal, prices_SPY], keys=['Portfolio', 'SPY'], axis=1)
        plot_normalized_data(df_temp,'Daily portfolio value and SPY',\
                             'date', 'normalized price')
#        ( df_temp / df_temp.values[0,:] ).plot()

    # Add code here to properly compute end value
    ev = portVal[-1]

    return cr, adr, sddr, sr, ev

def test_code():
    # This code WILL NOT be tested by the auto grader
    # It is only here to help you set up and test your code

    # Define input parameters
    # Note that ALL of these values will be set to different values by
    # the autograder!
    start_date = dt.datetime(2010,1,1)
    end_date = dt.datetime(2010,12,31)
    symbols = ['GOOG', 'AAPL', 'GLD', 'XOM']
    allocations = [0.2, 0.3, 0.4, 0.1]
    start_val = 1000000  
    risk_free_rate = 0.0
    sample_freq = 252

    # Assess the portfolio
    cr, adr, sddr, sr, ev = assess_portfolio(sd = start_date, ed = end_date,\
        syms = symbols, \
        allocs = allocations,\
        sv = start_val, \
        gen_plot = True)

    # Print statistics
    print "Start Date:", start_date
    print "End Date:", end_date
    print "Symbols:", symbols
    print "Allocations:", allocations
    print "Sharpe Ratio:", sr
    print "Volatility (stdev of daily returns):", sddr
    print "Average Daily Return:", adr
    print "Cumulative Return:", cr

if __name__ == "__main__":
    test_code()
