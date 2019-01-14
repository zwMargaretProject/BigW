import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA, ARMA
import statsmodels.api as sm
import itertools


class ARIMAEngine(object):
    def __init__(self):
        pass
    
    def __str__(self):
        return "This is a class for ARIMA time series analysis."
    
    def testStationarity(self, ts):
        dftest = adfuller(ts)
        dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
        for key, value in dftest[4].items():
            dfoutput['Critical Value ({})'.format(key)] = value
        print('*' * 20)
        print(dfoutput)
    
    def draw_acf_pacf(self, ts, lags=31):
        f = plt.figure(facecolor='white', figsize=(8,7))
        ax1 = f.add_subplot(211)
        plot_acf(ts, lags=lags, ax=ax1)
        ax2 = f.add_subplot(212)
        plot_pacf(ts, lags=lags, ax=ax2)
        plt.show()
    
    def seasonal(self, ts):
        decomposition = seasonal_decompose(ts, model='additive')
        trend = decomposition.trend
        seasonal = decomposition.seasonal
        residual = decomposition.resid
        return trend, seasonal, residual

    def draw_seasonal(self, ts):
        trend, seasonal, residual = self.seasonal(ts)
        plt.figure(figsize=(17,7), facecolor='white')
        plt.subplot(2,2,1)
        plt.plot(ts)
        plt.title('Original')
        plt.subplot(2,2,2)
        plt.plot(trend)
        plt.title('Trend')
        plt.subplot(2,2,3)
        plt.plot(seasonal)
        plt.title('Seasonal')
        plt.subplot(2,2,4)
        plt.title('Residual')
        plt.plot(residual)
        plt.show()
        
    def quality_test(self, model, bdq=(1,0,1)):
        print('*' * 20)
        print('DW Test')
        print(sm.stats.durbin_watson(model.resid.values))
        print('*' * 20)
        self.draw_acf_pacf(model.resid.values.squeeze(), lags=40)
        
    def draw_ts(self, timeSeries, title=None):
        plt.figure(facecolor='white', figsize=(12,5))
        plt.plot(timeSeries)
        plt.grid(True)
        if title is not None:
            plt.title(title)
        plt.show()
    
    def draw_multi_ts(self, ts_list, legend_list=None):
        f = plt.figure(facecolor='white', figsize=(12,5))
        for ts in ts_list:
            ts.plot(lw=2.5)
        if legend_list is not None:
            plt.legend(legend_list)
        plt.show()
  
    def proper_model(self, ts, maxLag=1):
        # Generate all different combinations of p, q and q triplets
        pdq = [(p, d, q) for p in range(0, maxLag+1) for d in range(0, maxLag+1) for q in range(0,maxLag+1)]

        min_param = None
        min_aic = None
        for param in pdq:
            seasonal_pdq = [(x[0], x[1], x[2], 12) for x in pdq]
            for param_seasonal in seasonal_pdq:
                try:
                    mod = sm.tsa.statespace.SARIMAX(ts,order=param,seasonal_order=param_seasonal,enforce_stationarity=False,enforce_invertibility=False)
                    results = mod.fit()
                    print('*' * 20)
                    print('AIC:{}, p-lab:{}, d-lag:{}, q-lag:{}, seasonal-lag:{}'.format(results.aic,param_seasonal[0], param_seasonal[1], param_seasonal[2], param_seasonal[3]))
                    if min_aic is None:
                        min_aic = results.aic
                        min_param = param_seasonal
                        proper_model = results
                    else:
                        if results.aic < min_aic:
                            min_aic = results.aic
                            min_param = param_seasonal
                            proper_model = results
                except:
                    continue
        print('*' * 20)
        print('Min AIC:{}, p-lab:{}, d-lag:{}, q-lag:{}, seasonal-lag:{}'.format(min_aic, min_param[0], min_param[1], min_param[2], min_param[3]))
        return proper_model

    def diag(self, proper_model):
        proper_model.plot_diagnostics(figsize=(15, 12))
        plt.show()
    
    def forcast(self, original_ts, proper_model, steps_forward=24 ):
        results = proper_model
        pred_uc = results.get_forecast(steps=steps_forward)
        # Get confidence intervals of forecasts
        pred_ci = pred_uc.conf_int()
        plt.title("Forecast",fontsize=15,color="red")
        ax = original_ts.plot(label='observed', figsize=(12, 5))
        pred_uc.predicted_mean.plot(ax=ax, label='Forecast', lw=2.5)
        ax.fill_between(pred_ci.index,
                        pred_ci.iloc[:, 0],
                        pred_ci.iloc[:, 1], color='k', alpha=.25)
        ax.set_xlabel('Date',fontsize=15)
        plt.legend()
        plt.show()
        return pred_uc