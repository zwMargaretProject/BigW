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
import sys
sys.path.append('C:/Users/acer/Desktop/BigW')

from timeSeriesAnalysis.arima import ARIMAEngine

if __name__ == '__main__':
    data = pd.read_table('D:/Cheese_Production_Data.txt', header=0, sep=',')
    months_dict = {'Jan':'01', 'Feb':'02', 'Mar':'03', 'Apr':'04', 'May':'05', 'Jun':'06', 'Jul':'07', 'Aug':'08', 'Sep':'09', 'Oct':'10', 'Nov':'11', 'Dec':'12'}
    years = data['Year']
    months = data['Month']
    dates = []
    for i in range(len(data)):
        dates.append('{}{}01'.format(years[i], months_dict[months[i]]))
    data['Date'] = pd.to_datetime(pd.Series(dates, index=data.index))
    data = data.set_index('Date')
    full_months = ['{}0{}01'.format(year, month) if month < 10 else '{}{}01'.format(year, month) for year in range(1995, 2014) for month in range(1, 13)]
    full_months = pd.Series(full_months)
    full_months = pd.to_datetime(full_months)
    df = pd.DataFrame(index=full_months)
    columns = list(data.columns)
    df[data.columns] = data[columns]
    df = df.fillna(method='ffill')
    df.index = pd.DatetimeIndex(df.index.values, freq=df.index.inferred_freq)
    engine = ARIMAEngine()
    for c in ['Cheese1.Prod', 'Cheese2.Prod', 'Cheese3.Prod']:
        ts = df[c]
        print('*---' * 20)
        engine.draw_ts(ts, c)
        engine.draw_seasonal(ts)
        engine.testStationarity(ts)
        model = engine.proper_model(ts)
        engine.diag(model)
        predict = engine.forcast(ts, model)