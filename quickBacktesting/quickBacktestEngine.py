import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from .quickPortfolio import *

class QuickBacktestEngine(object):
    def __init__(self):
    	pass
    
    def runBacktesting(self, symbol, bars, signals, initial_capital=100000.0):
    	# Create a portfolio of AAPL, with $100,000 initial capital
        portfolioEngine = MarketOnClosePortfolio()
        positions = portfolioEngine.generate_positions(signals)
        returns = portfolioEngine.backtest_portfolio(positions, bars, initial_capital)
        return returns

class MarketOnClosePortfolio(QuickPortfolio):

    def __init__(self):
        pass        
        
    def generate_positions(self, signals):
        positions = pd.DataFrame(index = signals.index).fillna(0.0)
        positions['signal'] = 100 * signals['signal']   # This strategy buys 100 shares
        return positions
                    
    def backtest_portfolio(self, positions, bars, initial_capital):
        portfolio = pd.DataFrame()
        portfolio['signal'] = positions['signal']
        portfolio['position_diff'] = positions.diff()
        portfolio['close'] = bars['Close']
        portfolio['holdings'] = positions['signal'] * bars['Close']
        portfolio['cash'] = initial_capital - (portfolio['position_diff'] * portfolio['close']).cumsum()
        portfolio['total'] = portfolio['cash'] + portfolio['holdings']
        portfolio['portfolio_returns'] =  portfolio['total'].pct_change()
        return portfolio
     