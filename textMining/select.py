import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime,timedelta
import os
from datetime import datetime,date


###########################################################
def collectStockName(filepath):
    '''
    Collect stock symbols with price data in specific dirs.
    
    Args:
        filepath(str): Filepath from which we want to collect stock tickers.
        
    Returns:
        stock_list(list): List of tickers that have CSV in specific filepath.
    '''
    stock_list=[]
    for root, dirs, files in os.walk(filepath):
        if files:
            for f in files:
                if 'csv' in f:
                    stock_list.append(f.split('.csv')[0])
    return stock_list


=================================================================


def getUniqueDate(data,days):
    '''
    For dates covering same event windows, select the earlist dates.

    Args:
        data(DataFrame): data
        days(int): length of event windows, like 11 for CAR1 or 21 for CAR2
    
    Returns:
        data(DataFrame): data with earlist dates for no-lapping event windows
    '''

   
    data['Date'] = pd.to_datetime(data['Date'])
    data=data.set_index('Date')

    dates =data.index
    unique_dates = []
    unique_dates.append(dates[0])

    window_length = pd.to_timedelta(days,unit='d')

    if len(dates) >= 2:
        for i in dates[1:]:
            window_start = unique_dates[-1]
            window_end = window_start + window_length
            if i > window_end:
                unique_dates.append(i)

    unique_datas = np.array(unique_dates)
    data = data[data.index.isin(unique_dates)]

    return data

========================================================

def concatAllData(filepath):
    '''
    Concat all data together for CSVs in same filepath.

    Args:
        filepath(str): filepath from which data needs to be concated

    Returns:
        output(DataFrame): output of all data
    '''
    output = pd.DataFrame()
    ticker_list = collectStockName(filepath)
    for ticker in ticker_list:
        data = pd.read_csv(filepath+ticker+'.csv')
        data['Symbol'] = ticker
        output = pd.concat([output,data],ignore_index=True)

    return output.set_index('Symbol')



========================================================
def carSelect(filepath,filepath_output,car_column='CAR[-W:W]',t_column='CAR[-W:W] t-Score',car_criteria=0.2,t_criteria=1.96,days=21):

    '''
    Select data with significantly high data and output the data.

    Args:
        filepath(str): filepath in which raw data is stored in
        filepath_output(str): filepath to which results are stored in 
        car_column(str): For CAR1, car_column = 'CAR[-W:0]'
                         For CAR2, car_column = 'CAR[-W:W]'

        t_column(str): For CAR1, t_column = 'CAR[-W:0] t-Score'
                       For CAR2, t_column = 'CAR[-W:W] t-Score'
        car_criteria(float): For CAR1, car_criteria = 0.1
                             For CAR2, car_criteria = 0.2
        
        t_criteria(float): t-statistics for 95% significance level is 1.96

    Returns:
        DataFrame: all data with significantly high CAR
    
    
    '''
    ticker_list = collectStockName(filepath)

    for ticker in ticker_list:
        
        data = pd.read_csv(filepath+ticker+'.csv')
 
        data = data[['Date',car_column,t_column]]
        data = data[data[car_column]>=car_criteria]
        data = data[data[t_column]>=t_criteria]

        data = getUniqueDate(data)

        data.to_csv(filepath_output+ticker+'.csv')
    
    print('Output Done')

    print('Begin to concat all data')

    return concatAllData(filepath_output)

