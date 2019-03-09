import os
import pandas as pd
import datetime
import numpy as np


###############################
def collectFilename(filepath,filetype='csv'):
    '''
    Collect names of files in specific dirs.
    
    Args:
        filepath(str): Filepath we want to collect stock tickers.
        
    Returns:
        file_list(list): List of tickers that have CSV in specific filepath.
    '''
    filename_list=[]
    for root, dirs, files in os.walk(filepath):
        if files:
            for f in files:
                if filetype in f:
                    filename_list.append(f.split('.'+filetype)[0])
    return filename_list

###########################       
def week_number(df):   
    '''
    Get the week number according to datetime.

    Args:
    df(pd.DataFrame): dataframe that includes 'Date' data.
    '''
    df['year'] = df['Date'].dt.year
    df['month'] = df['Date'].dt.month
    df['day'] = df['Date'].dt.day    
    year_list = list(df['year'])
    month_list = list(df['month'])
    day_list = list(df['day'])
    week_num_list = []
    for i in range(len(df)):
        a,week_num,b = datetime.date(int(year_list[i]),int(month_list[i]),int(day_list[i])).isocalendar()
        week_num_list.append(week_num)
    df['week_num'] = pd.Series(week_num_list,index=df.index)   
    return df



################################
def simple_returns(df,freq,week_gap=1):  
    '''
    Compute simple returns.
    
    Args:
    df(pd.DataFrame): A dataframe that includes close prices.
    freq(str): 'D'(daily) or 'W'(weekly) or 'M'(monthly) or 'A'(annual).
    week_gap(int): If returns are for one week, week_gap would be 1.
                   If returns are for two weeks' returns, week_gap would be 2.

    '''
    
    df = df.resample(freq).last()
    df['next_week_return'] = df['Close'].shift(-week_gap)/df['Close']-1
    return df


##################################      
def select_years(df,start_year):
    df = df[df['year'] >= start_year]
    return df

##################################  
def multi_simple_returns(filepath_csv,
                         filepath_output,
                         freq='W',
                         start_year = None,
                         get_week_num = True,
                         iex = False,
                         price = 1,
                         week_gap = 1):        
    ''' Compute simple returns and get week num for all stocks'''
    
    filename_list = collectFilename(filepath_csv,'csv')
    x = 0
    for symbol in filename_list:
        x+=1
        print(x)
        print(symbol)
        
        filename_csv = filepath_csv+symbol+'.csv'
        filename_output = filepath_output + symbol + '.csv'        
        df = pd.read_csv(filename_csv)
        
        
        # columns of stock prices from IEX database is different from columns of prices from Yahoo!Finance.
        # Hence, it is neccessary to make columns remain the same.
        if iex == True:
            df = df.rename(columns={'date':'Date','open':'Open','high':'High','low':'Low','close':'Close','volume':'Volume'})
        closes = list(df['Close'])

        all_high = True

        i = 0
        while i < len(closes) and all_high:
            if closes[i] < price:
                all_high = False
            i += 1
        
        if all_high:
            df = df[['Date','Close','Volume']]
            df['Capital'] = df['Close']*df['Volume']
            df['Date'] = pd.to_datetime(df['Date'])
            if get_week_num == True:
                df = week_number(df)
                
            if start_year is not None:
                df = select_years(df,start_year)
                
            df = df.set_index('Date')
            output = simple_returns(df=df,freq=freq,week_gap=week_gap)
            output.to_csv(filename_output)
    print('All data with log returns has been saved to filepath: '+filepath_output)




###############################################3

def return_f():

    dir_prices_1 = 'D:/data/StockPrices/yahoo_all/'
    dir_prices_2 =  'D:/data/StockPrices/iex_all/'
    
   
    for year in [2015,2016,2017,2018]:
    
        filepath_output = 'D:/data/StockPrices/yahoo_iex_{}/'.format(year)
        
        
        
        multi_simple_returns(dir_prices_1,
                             filepath_output,
                             freq='W',
                             start_year = year,
                             get_week_num = True,
                             iex = False,
                             price = 1,
                             week_gap = 1)
        
        multi_simple_returns(dir_prices_2,
                             filepath_output,
                             freq='W',
                             start_year = year,
                             get_week_num = True,
                             iex = True,
                             price = 1,
                             week_gap = 1)


def idx_f():
    dir = 'D:/data/input/index/'
    
    for year in [2015,2016,2017,2018]:
        filepath_output = 'D:/data/input/index/returns_{}/'.format(year)
    
        multi_simple_returns(dir,
                             filepath_output,
                             freq='W',
                             start_year = year,
                             get_week_num = True,
                             iex = True,
                             price = 5,
                             week_gap = 1)


if __name__ == '__main__':
    return_f()
    idx_f()
    
   