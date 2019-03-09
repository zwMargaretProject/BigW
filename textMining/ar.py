import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime,timedelta,date
import os
import pandas_datareader.data as web


###########################################################
def collectStockName(filepath):
    '''
    Collect stock symbols with price data in specific dirs.
    
    Args:
        filepath(str): Filepath we want to collect stock tickers.
        
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



###########################################################
def loadData(ticker,filepath):
    '''
    Load CSV and get close price data.
    
    Args:
        ticker(str): Ticket of stock whose data is needed to be loaded to Pandas.
        filepath(str): Filepath to which stock data (CSV) is stored.
    
    Returns:
        data(DataFrame): DataFrame with "Close" values(closed prices) and " Date "index.
    
    '''

    # Convert types of data from "object" to "float32". This helps to speed up computation.
    data = pd.read_csv(filepath + ticker + '.csv',dtype={'Close':np.float32})
    data = data[['Date','Close']]
    data['Date'] = pd.to_datetime(data['Date'], format='%Y-%m-%d')
        
    # Set index as "Date"
    return data.set_index('Date')



def getLogReturns(df):
    '''
    Compute daily log returns for close prices. 
    Log Returns(t) = log(close price(t)) - log(close price(t-1))

    Args:
        df(DataFrame): DataFrame with "Close" values(closed prices) and " Date "index
    
    returns:
        df(DataFrame): DataFrame with index 'Date' and columns [['Close','Index Close','Log Returns','Index Log Returns']]

    '''
    df['Log Returns'] = np.log(df['Close']/df['Close'].shift(1))
    df['Index Log Returns'] = np.log(df['Index Close']/df['Index Close'].shift(1))
    return df
    

###########################################################
def dataSelect(df, event_date, L, W):
    '''
    Select data for [t-L-W, t+W] period when the event date is t.
    L=240 and W=10 in this case.

    Args:
        df(DataFrame): DataFrame with index 'Date' and columns [['Close','Index Close','Log Returns','Index Log Returns']]
        event_date(datetime): Date
        L(int): Date length for estimation period. 
        W(int): 2W+1 is date length for event window.
    
    Returns: 
        df(DataFrame): DataFrame covering [t-L-W, t+W] period
    '''
    dates = df.index
    
    n = len(dates)
    
    for i in range(n):
        if dates[i] == event_date:
            break
    
    start = dates[i-(L+W)]
    end = dates[i+W]
    
    return df.loc[start:end,:]

###################################################################
def linearRegression(x, y):
    '''
    Make OLS regression for x and y
    
    Args:
        x(array): X in OLS model
        y(array): Y in OLS model
    
    Returns:
        ahat(float): Intercept Alpha in OLS model
        bhat(float): Coefficient Beta in OLS model
        sigma2(float): Sample sum of residuals
        xbar(float): Mean of X
        dx2(float): Square sum of X

    '''
    xbar = np.mean(x)
    ybar = np.mean(y)
    dx = x - xbar
    dy = y - ybar

    dx2 = sum(dx*dx)

    bhat = sum(dx*dy)/dx2
    ahat = ybar - bhat*xbar

    n = len(x)
    residual = y - ahat - bhat*x
    sigma2 = sum(residual*residual)/(n-2)

    return ahat, bhat, sigma2, xbar, dx2



###########################################################

def getAbnormalReturns(df,L,W):
    '''
    Compute abnormal returns and cumulative abnormal returns for a single stock.
    The compuation would not be processed if data covers less than L+2W+1 days.
    Outputs include: Daily abnormal returns(AR) and t-statistics; 
                     Cumulative abnormal returns(CAR[-W:0]) for past W days and t-statistics;
                     Cumulative abnormal returns(CAR[-W:W]) for past W days and post W days and t-statistics.
    
    Args:
        df(DataFrame): DataFrame " Date "index and [['Close','Index Close','Log Returns','Index Log Returns']] columns
        L(int): Date length for estimation period. 
        W(int): 2W+1 is date length for event window.

    Returns:
        DataFrame with index of 'Date' 
        and columns [['AR','AR t-Score','CAR[-W:0]','CAR[-W:0] t-Score','CAR[-W:W]','CAR[-W:W] t-Score']]
    '''

    dates = df.index
    n = len(dates)
    
    # Create zero array for dates
    dates_array = dates[L+W+1:n-W]

    # Create zero array for abnormal returns(AR)
    ar_array = np.zeros(len(dates_array))
    # Create zero array for AR t-statistics 
    ar_t = np.zeros(len(dates_array))

    # Create zero array for CAR[-W:0]
    car1_array = np.zeros(len(dates_array))
    # Create zero array for CAR[-W:0] t-statistics
    car1_t = np.zeros(len(dates_array))

    # Create zero array for CAR[-W:W]
    car2_array = np.zeros(len(dates_array))
    # Create zero array for CAR[-W:W] t-statistics
    car2_t = np.zeros(len(dates_array))
    

    for i in np.arange(L+W+1,n-W):
        event_date = dates[i]

        # Get data covering [t-L-W-1,t+W] period
        data = dataSelect(df,event_date,L,W) 
        
        # Get stock logreturns array
        y = data['Log Returns'].values
        # Get index logreturns array
        x = data['Index Log Returns'].values
   
        # Make OLS regression for estimation period
        ahat, bhat, sigma2, rmbar, drm2sum = linearRegression(x[0:L],y[0:L])

        # Compute daily abnormal returns: AR = Actual stock return - alpha - beta * Index return
        ar_values = y[L:] - ahat - bhat*(x[L:])

        drm = x[L:]- rmbar
        s2 = sigma2*(1 + 1/L + drm*drm/drm2sum)
        
        # Compute t-statistics for AR, CAR[-W:0] and CAR[-W:W]
        ar = ar_values[W]
        ar_tstat = ar/np.sqrt(s2[W])

        car1 = ar_values[0:W+1].sum()
        carv1 = s2[0:W+1].sum()
        car_tstat1 = car1/np.sqrt(carv1)

        car2 = ar_values.sum()
        carv2 = s2.sum()
        car_tstat2 = car2/np.sqrt(carv2)
 
        # Insert values to arrays
        ar_array[i-L-W-1] = ar
        ar_t[i-L-W-1] = ar_tstat

        car1_array[i-L-W-1] = car1
        car1_t[i-L-W-1] = car_tstat1

        car2_array[i-L-W-1] = car2
        car2_t[i-L-W-1] = car_tstat2
    
    # Return a DataFrame
    return pd.DataFrame({'Date':dates_array,
                         'AR':ar_array,
                         'AR t-Score':ar_t,
                         'CAR[-W:0]':car1_array,
                         'CAR[-W:0] t-Score':car1_t,
                         'CAR[-W:W]':car2_array,
                         'CAR[-W:W] t-Score':car2_t}).set_index('Date')



###########################################################

def outputResults(ticker_list,filepath_ticker,filepath_index,filepath_output,index_name='SPY',L=240,W=10):
    '''
    Combine all functions together. This function can be used directly to output all results.
    
    Args:
        ticker_list(list): List of tickers whose data was originally planned to download
        filepath_index(str): Filepath of CSV which includes daily data of market index
        filepath_output(str): Filepath to which output data is stored.
        index_name(str): Name of index. CSV file of index data is 'SPY.csv' in this case.
        L(int): Date length for estimation period. 
        W(int): 2W+1 is date length for event window.
    '''
    
    # Load market index data
    index_df = loadData(index_name,filepath_index)

    for ticker in ticker_list:
        
        # Load stock data
        df = loadData(ticker,filepath_ticker)
        
        # If stock data covers less than L+2W+1 days, pass
        if len(df)>(2*W+L+1):

            # Merge stock data and index data together
            df['Index Close'] = index_df['Close']
            # Compute log returns for stock prices and index prices
            df = getLogReturns(df)

            # Compute AR and CAR, then reserve outputs CSV to specific filepath
            getAbnormalReturns(df,L,W).to_csv(filepath_output+ticker+'.csv')
    print ('ALL DONE!')


#======================================================================

def main():

    filepath_index = '/input/'

    # Please change filepaths here.
    
    #--------------------------------------------------

    # filepath_ticker_1 is where stock prices from Yahoo! Finance are saved in.
    filepath_ticker_1 = '/output/StockPrices/StockPrices_YahooFinance/'

    # filepath_output_1 is where abnormal returns for stocks from Yahoo! Finance need to be saved in.
    filepath_output_1 = '/output/AbnormalReturns/AR_YahooFinance/'

    # ticker_list_1 is to collect stock names whose data is saved in filepath_ticker_1
    ticker_list_1 = collectStockName(filepath_ticker_1)

     # Calculate abnormal returns for stocks from Yahoo! Finance DataSet
    outputResults(ticker_list_1,filepath_index,filepath_output_1)

    #---------------------------------------------------------------------
    filepath_ticker_2 = '/data/output/StockPrices/StockPrices_IEX/'
    filepath_output_2 = '/data/output/AbnormalReturns/AR_IEX/'
    ticker_list_2 = collectStockName(filepath_ticker_2)

    outputResults(ticker_list_2,filepath_index,filepath_output_2)



if __name__ == '__main__':
    main()

   

    
