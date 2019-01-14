
import arch
from scipy import  stats
import statsmodels.api as sm  
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import fix_yahoo_finance as yf
from datetime import date, datetime
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA, ARMA
import statsmodels.api as sm
import itertools

def yahooFinanceDownload(ticker, start_date= datetime(2000,1,1), end_date = date.today()):
    prices = yf.download(ticker, start=start_date, end=end_date)
    
    # If no data from Yahoo! Finance, the lenth of output will be zero
    if len(prices):
        return prices
    else:
        print ('Not available')


class ARCHAndGARCH(object):
    def __init__(self, arma_model):
        self.ts = ts
        
    def model_fit(self, ar_lag, vol_lag, type='ARCH'):
        model = arch.arch_model(self.ts, mean='AR', lags=ar_lag, vol=type, p=vol_lag)
        res = model.fit()
        return res
    
    def get_summary(self, res):
        print(res.summary)
        print(res.params)


def main():
    engine = ARIMAEngine()
    ### Download Data
    sti = yahooFinanceDownload(ticker='STI',start_date=datetime(2018,1,1), end_date =datetime(2018,12,31))['Adj Close']
    ts = np.log(sti/sti.shift(1)).dropna()

    ### ARCH
    e = ARCHAndGARCH(ts)
    res = e.model_fit(0, 1, 'arch')
    e.get_summary(res)
    
    ### GARCH
    res2 = e.model_fit(0,1,'GARCH')
    e.get_summary(res2)

if __name__=='__main__':
    main()
    

