
import numpy as np
import pandas as pd
import fix_yahoo_finance as yf
from datetime import datetime, date
import matplotlib.pyplot as plt 
%matplotlib inline
import scipy.optimize as sco

class EfficientFrontier(object):
	def __init__(self, ticker_list, rf):
		self.ticker_list = ticker_list
		self.no_assets = len(ticker_list)
		self.rf = rf
		self.rets = None
        
	def yahoo_finance_download(self):
		merges = pd.DataFrame()
		for ticker in self.ticker_list:
			prices = yf.download(ticker, start=datetime(2017,1,1), end=date.today())
			if len(prices):
				merges[ticker] = prices['Adj Close']
		self.rets = np.log(merges / merges.shift(1))

	def returns_statistics(self):
		print('Mean')
		print(self.rets.mean())
		print('Volatility')
		print(np.sqrt(self.rets.var()))
		print('Covariance')
		print(self.rets.cov())

	def monte_carlo(self, num=10000):
		prets = []
		pvols = []
		no_assets = self.no_assets
		rets = self.rets
		for p in range(num):
			weights = np.random.random(no_assets)
			weights /= np.sum(weights)
			prets.append(np.sum(rets.mean() * weights) * 252)
			pvols.append(np.sqrt(np.dot(weights.T, np.dot(rets.cov() * 252, weights))))
		return np.array(prets), np.array(pvols)

	def statistics(self, weights):
		rets, rf = self.rets, self.rf
		weights = np.array(weights)
		pret = np.sum(rets.mean() * weights) * 252
		pvol = np.sqrt(np.dot(weights.T, np.dot(rets.cov() * 252, weights)))
		return np.array([pret, pvol, (pret - rf) / pvol])

	def min_func_sharpe(self, weights):
		return -self.statistics(weights)[2]

	def get_max_sr_weights(self, weights_sum, bound1, bound2):
		no_assets = self.no_assets
		cons = ({'type':'eq', 'fun': lambda x : np.sum(x) - weights_sum})
		bnds = tuple((bound1,bound2) for x in range(no_assets))
		init = no_assets * [1 / no_assets]
		opts = sco.minimize(self.min_func_sharpe, init, method='SLSQP', bounds=bnds, constraints=cons)
		return opts['x'].round(3)

	def plot_monte_carlo(self, pvols, prets):
		plt.figure(figsize=(8,4))
		plt.scatter(pvols, prets, c=(prets-self.rf)/pvols, marker='o')
		plt.grid(True)
		plt.xlabel('Volatility')
		plt.ylabel('Return')
		plt.colorbar(label='Sharpe Ratio')
	
	def min_func_port(self, weights):
		return -self.statistics(weights)[1]
	
	def efficient_frontier(self, weights_sum, bound1, bound2):
		trets = np.linspace(0.0, 0.25, 50)
		tvols = []
		bnds = tuple((bound1, bound2) for x in range(self.no_assets))
		init = self.no_assets * [1 / self.no_assets]
		for tret in trets:
			cons = ({'type':'eq', 'fun':lambda x: self.statistics(x)[0] - tret}, {'type':'eq', 'fun': lambda x: np.sum(x) - weights_sum})
			res = sco.minimize(self.min_func_port, init, method='SLSQP', bounds=bnds, constraints=cons)
			tvols.append(res['fun'])
		tvols=np.array(tvols)
		return tvols, trets

def main():
    # Download data from Yahoo! Finance
    ef = EfficientFrontier(ticker_list=['AAPL', 'IBM', 'FB'], rf=0.01)
    ef.yahoo_finance_download()
    ef.returns_statistics()
    
    # plot monte carlo points
    prets, pvols = ef.monte_carlo()
    ef.plot_monte_carlo(pvols, prets)
    
    setting = [(1.000, -1.000, 1.000), (1.000, 0.300, 0.500), (0.000, -0.400, 0.400)]
    for s in setting:
        weights = ef.get_max_sr_weights(s[0], s[1], s[2])
        ef.print_results(weights, s)

if __name__ == ‘__main__’:
    main()
