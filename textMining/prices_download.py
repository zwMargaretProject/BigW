import pandas as pd
import pandas_datareader.data as web
from datetime import datetime, timedelta, date
from pandas_datareader._utils import RemoteDataError
import os
import fix_yahoo_finance as yf

#============================================================

def getTickerList(ticker_file):
    '''
    Get list of tickers (stock symbols) from companylist.csv.   
    Args:
        ticker_file(str): Filepath of CSV file that includes stock symbols
    returns:
        list: The symbols
            '''
    return pd.read_csv(ticker_file)['Symbol']


#============================================================

def yahooFinanceDownload(ticker,filepath,start_date,end_date):
    '''Download daily stock prices for a single stock from Yahoo! Finance and reserve data as CSV to specific filepath.
    Args:
        ticker(str): Ticker
        filepath(str): Filepath to output and reserve CSV
    '''
    try:
        prices = yf.download(ticker, start=start_date, end=end_date)
    except:
        pass
    else:    
        # If no data from Yahoo! Finance, the lenth of output will be zero
        if len(prices):
            out_filename = filepath + ticker +  '.csv'
            prices.to_csv(out_filename)
#============================================================

def yahooFinanceMultiple(ticker_list,filepath,start_date=datetime(2000,1,1),end_date=date.today()):
    '''
    Download data for a series of stocks from Yahoo! Finance and reserve data as CSVs in to specific filepath.
    
    Args:
        ticker_list(list): A list of tickers
        filepath(str): Filepath to output and reserve CSVs
    '''
    for ticker in ticker_list:
        yahooFinanceDownload(ticker,filepath,start_date,end_date)


#============================================================

def quandlDownload(ticker,filepath):
    '''Download daily stock prices for a single stock from Quandl and reserve data as CSV to specific filepath.

    Args:
        ticker(str): Ticker 
        filepath(str): Filepath to output and reserve CSV
    '''
    start = datetime(2000,1,1)
    end = date.today()
    prices = web.DataReader(ticker, 'quandl', start, end)
    out_filename = filepath + ticker +  '.csv'
    prices.to_csv(out_filename)

#============================================================

def iexDownload(ticker,filepath,start_date,end_date):
    '''Download daily stock prices for a single stock from IEX and reserve data to specific filepath.

    ***Attention: IEX provides data for only recent 5 years
       Hence,the start date of downloading data is set to be 2013-06-01. 

    Args:
        ticker(str): Ticker 
        filepath(str): Filepath to output and reserve CSV
    '''
    prices = web.DataReader(ticker, 'iex', start_date, end_date)
    out_filename = filepath + ticker +  '.csv'
    prices.to_csv(out_filename)

#============================================================
def iexMultiple(ticker_list,filepath,start_date,end_date):
    b=0
 
    for ticker in ticker_list:
        b += 1
        try:
            iexDownload(ticker,filepath,start_date,end_date)
            print(b)
            print(ticker+'  ===')

        # If no data from IEX, KeyError or FileNotFoundError would be raised
        except (KeyError,FileNotFoundError):
            pass


#============================================================
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

#============================================================
def getDownloadList(ticker_list,filepath):
    '''
    Get stock symbols that have been successfully downloaded.
    
    Args:
        ticker_list(list): List of tickers whose data was originally planned to download.
        filepath(str): Filepath to which data has been stored.
    
    Returns:
        download_list(list): List of tickers whose data has been successfully downloaded and stored.
        unsuccess_list(list): List of tickers whose data has been unsuccessfully dowloaded or stored.
    '''

    download_list = collectStockName(filepath)
    unsuccess_list = [ticker for ticker in ticker_list if ticker not in download_list]
    return download_list, unsuccess_list
    

#============================================================
def download_all_prices(ticker_list,filepath_1,filepath_2,start_date=datetime(2000,1,1),end_date=date.today()):
    '''
    Download stock prices.
    First, try to download prices from Yahoo!Finance.
    Then, for a specific stock, if there is no data available from Yahoo!Finance, turn to IEX.
    
    Args:
        ticker_list: list
        filepath_1:  str, filepath for saving prices from Yahoo!Finance
        filepath_2:  str, filepath for saving prices from IEX
        start_date:  datetime
        end_date:    datetime
    '''
    for ticker in ticker_list:
        yahooFinanceDownload(ticker,filepath_1,start_date,end_date)
    yahoo_list, iex_list = getDownloadList(ticker_list,filepath_1)
    iexMultiple(iex_list,filepath_2,start_date,end_date)
    print('=== Stock Prices have been downloaded from Yahoo! Finance and IEX. ===')
