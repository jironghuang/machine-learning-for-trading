"""MC1-P2: Optimize a portfolio."""

import pandas as pd
import numpy as np
import scipy.optimize as spo
import datetime as dt
from util import get_data

import sys
sys.path.insert(0, '../mc1_p1')
from analysis import get_portfolio_value, get_portfolio_stats, plot_normalized_data

def cost(allocs, prices, start_val, daily_rf, samples_per_year):
    portVal = get_portfolio_value(prices, allocs, start_val)
    cr, adr, sddr, sr = get_portfolio_stats(portVal, daily_rf, samples_per_year)
    return -sr

    
# This is the function that will be tested by the autograder
# The student must update this code to properly implement the functionality
def optimize_portfolio(sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,1,1), \
    syms=['GOOG','AAPL','GLD','XOM'], gen_plot=False):
    start_val = 1000000
    daily_rf = 0
    samples_per_year = 252
    # Read in adjusted closing prices for given symbols, date range
    dates = pd.date_range(sd, ed)
    prices_all = get_data(syms, dates)  # automatically adds SPY
    prices = prices_all[syms]  # only portfolio symbols
    prices_SPY = prices_all['SPY']  # only SPY, for comparison later

    # find the allocations for the optimal portfolio
    allocGuess = np.ones(len(syms), dtype = 'float32') / len(syms)
    setBnds = tuple( [(0,1) for x,y in enumerate(allocGuess)] )#create tuple of (0,1) tuples
    # 'constraints' below constrains allocations to sum to 1 
    # and 'setBnds' forces each allocation to lie in (0,1)
    srMax = spo.minimize(cost, allocGuess, bounds = setBnds, \
            constraints = ({ 'type': 'eq', 'fun': lambda inputs: 1.0 - np.sum(inputs) }), \
            args = (prices,start_val,daily_rf,samples_per_year,), \
            method = 'SLSQP', options = {'disp': True})
    allocs = srMax.x
    
    portVal = get_portfolio_value(prices, allocs, start_val)
    cr, adr, sddr, sr = get_portfolio_stats(portVal, daily_rf, samples_per_year)

    # Compare daily portfolio value with SPY using a normalized plot
    if gen_plot:
        df_temp = pd.concat([portVal, prices_SPY], keys=['Portfolio', 'SPY'], axis=1)
        plot_normalized_data(df_temp, 'Optimized portfolio values', 'date', \
                             'normalized price')

    return allocs, cr, adr, sddr, sr

    
def test_code():
    # This function WILL NOT be called by the auto grader
    # Do not assume that any variables defined here are available to your function/code
    # It is only here to help you set up and test your code

    # Define input parameters
    # Note that ALL of these values will be set to different values by
    # the autograder!

    start_date = dt.datetime(2005,12,1)
    end_date = dt.datetime(2006,5,31)
    symbols = ['YHOO', 'HPQ', 'GLD', 'HNZ']

    # Assess the portfolio
    allocations, cr, adr, sddr, sr = optimize_portfolio(sd = start_date, ed = end_date,\
        syms = symbols, \
        gen_plot = True)

    # Print statistics
    print "Start Date:", start_date
    print "End Date:", end_date
    print "Symbols:", symbols
    print "Allocations:", allocations
    print "sum allocations:", allocations.sum()
    print "Sharpe Ratio:", sr
    print "Volatility (stdev of daily returns):", sddr
    print "Average Daily Return:", adr
    print "Cumulative Return:", cr

if __name__ == "__main__":
    # This code WILL NOT be called by the auto grader
    # Do not assume that it will be called
    test_code()
