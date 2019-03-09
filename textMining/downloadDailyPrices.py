
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

def yahooFinanceDownload(ticker,filepath):
    '''Download daily stock prices for a single stock from Yahoo! Finance and reserve data as CSV to specific filepath.

    Args:
        ticker(str): Ticker
        filepath(str): Filepath to output and reserve CSV
    '''
    start_date= datetime(2000,1,1)
    end_date = date.today()
    prices = yf.download(ticker, start=start_date, end=end_date)
    
    # If no data from Yahoo! Finance, the lenth of output will be zero
    if len(prices):
        out_filename = filepath + ticker +  '.csv'
        prices.to_csv(out_filename)


def yahooFinanceMultiple(ticker_list,filepath):
    '''
    Download data for a series of stocks from Yahoo! Finance and reserve data as CSVs in to specific filepath.
    
    Args:
        ticker_list(list): A list of tickers
        filepath(str): Filepath to output and reserve CSVs
    '''
    for ticker in ticker_list:
        yahooFinanceDownload(ticker,filepath)


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



def quandlMultiple(ticker_list,filepath):
     '''Download daily stock prices for multiple stocks from Quandl and reserve data as CSVs to specific filepath.

    Args:
        ticker_list(list): List of tickers
        filepath(str): Filepath to output and reserve CSVs
    '''
    start = datetime(2000,1,1)
    for ticker in ticker_list:
        try:
            quandlDownload(ticker,filepath)

        # If no data from Quandl, RemoteDataError would be raised
        except RemoteDataError: 
            pass



#============================================================

def iexDownload(ticker,filepath):
    '''Download daily stock prices for a single stock from IEX and reserve data to specific filepath.

    ***Attention: IEX provides data for only recent 5 years
       Hence,the start date of downloading data is set to be 2013-06-01. 

    Args:
        ticker(str): Ticker 
        filepath(str): Filepath to output and reserve CSV
    '''

    start = datetime(2013,6,1)
    end = date.today()
    prices = web.DataReader(ticker, 'iex', start, end)
    out_filename = filepath + ticker +  '.csv'
    prices.to_csv(out_filename)


def iexMultiple(ticker_list,filepath):
     '''Download daily stock prices for multiple stocks from IEX and reserve data as CSVs to specific filepath.

    *** Attention: IEX provides data for only recent 5 years.
    Args:
        ticker_list(list): List of tickers
        filepath(str): Filepath to output and reserve CSVs
    '''
    for ticker in ticker_list:
        try:
            iexDownload(ticker,filepath)

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
    



#------------------------------------------------------

def main():
    '''
    
    filepath_output_1: filepath where stock prices from Yahoo! Finance are saved.
    filepath_output_2: filepath where stock prices from IEX are saved.
                       For stocks with no data in Yahoo! Finance, the daily data would be downloaded from IEX.
    '''

    # Please change filepaths before running the codes.
    ticker_file = '/data/input/companylist.csv'
    filepath_output_1 = '/data/output/StockPrices/StockPrices_YahooFinance/'
    filepath_output_2 = '/data/output/StockPrices/StockPrices_IEX/'

    # get list of all tickers
    ticker_list = getTickerList(ticker_file)
    
    # download daily prices from Yahoo! Finance first
    yahooFinanceMultiple(ticker_list,filepath_output_1)
    
    # get list of tickers with no data in Yahoo! Finance
    download_list, unsuccess_list = getDownloadList(ticker_list,filepath_output_1)

    # download daily prices for rest tickers from IEX
    iexMultiple(unsuccess_list,filepath_output_2)


if __name__ == '__main__':
    main()


    





























