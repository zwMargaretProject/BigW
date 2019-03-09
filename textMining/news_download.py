import pandas as pd
import urllib
import json
import time
import os
import socket
from general_functions import collectFilenames,check_and_create_dir

###############################
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

###############################
def saveContentsAsJson(contents,filepath,filename):
    with open(filepath+filename+'.json','w',encoding='utf-8') as json_file:
        json.dump(contents,json_file,ensure_ascii=False)
###############################        
def loadJson(filepath,filename):
    with open(filepath+filename+'.json','r',encoding='utf-8') as json_file:
        contents=json.load(json_file)
    return contents
  
###############################
def downloadNewsFromApi(stock_id,start_date,end_date,api_link,time_zone):
    
    stock_id = str(int(stock_id))
    start_date = str(start_date)[:10]
    end_date = str(end_date)[:10]

    full_link = api_link+stock_id+'&&start_date='+end_date+'&&end_date='+start_date

    if time_zone:
        full_link = full_link+'&timezone='+time_zone

    try:
        socket.setdefaulttimeout(20)
        fh = urllib.request.urlopen(full_link)
        data = fh.read()
        news = json.loads(str(data, encoding='utf-8'))
    except:
        news = None
        print('Time Out')
        pass

    return news

###############################
def news_for_year(stock_id,year,api_link,time_zone):
    start_date = str(year) + '-01-01'
    end_date = str(year) + '-12-31'
    news = downloadNewsFromApi(stock_id,start_date,end_date,api_link,time_zone)
    return news

###############################
def news_function(output_dir, data_df, year, api_link, time_zone):
    for i in range(len(data_df)):
        print(i+1)
        stock_id = data_df.iloc[i]['company_id']
        stock_name = data_df.iloc[i]['Symbol']
        news = news_for_year(stock_id,year,api_link,time_zone)

        if news is not None and len(news['data']) != 0 and news['data'] != 0:
            filename = stock_name
            saveContentsAsJson(news,output_dir,filename)
            
            print(stock_name + '     Down --------------')
            
        else:
            print(stock_name + ': No news for year: {}'.format(str(year)))
    
        time.sleep(2)
        print('  Sleep Finished ')
        
###############################
def news_for_abnormal_samples(output_dir, car_df, year, api_link, time_zone,start = 0):
    data_df = car_df[car_df['year'] == year]
    for i in range(start,len(data_df)):
        print(i+1)
        stock_id = data_df.iloc[i]['company_id']
        stock_name = data_df.iloc[i]['Symbol']
        start_date = data_df.iloc[i]['Start_Date']
        end_date = data_df.iloc[i]['End_Date']
        event_date = data_df.iloc[i]['Date']
        news = downloadNewsFromApi(stock_id,start_date,end_date,api_link,time_zone)

        if news is not None and len(news['data']) != 0 and news['data'] != 0:
            filename = stock_name + '_' + str(event_date)[:10]
            saveContentsAsJson(news,output_dir,filename)
            
            print(stock_name + '     Down --------------')
            
        else:
            print(stock_name + ': No news for year: {}'.format(str(year)))
    
        time.sleep(2)
        print('  Sleep Finished ')


        
    