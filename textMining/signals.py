import os
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt

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
        
def week_number(df):
    df['Date'] = pd.to_datetime(df['Date'])
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

def simple_returns(df,freq,length_period):    

    df = df.resample(freq).first()

    for n in range(length_period):
        df['simple_return_'+str(n+1)] = (df['close'].shift(-n)/df['close']-1)/(n+1)

    return df
        

def simple_returns_for_multi_df(filepath_csv,filepath_output,freq='W',length_period=10):

    filename_list = collectFilename(filepath_csv,'csv')

    for symbol in filename_list:
        filename_csv = filepath_csv+symbol+'.csv'
        filename_output = filepath_output + symbol + '.csv'

        simple_returns(filename_csv,freq,length_period).to_csv(filename_output)

    print('All data with log returns has been saved to filepath: '+filepath_output)


#------------------------------------------

def delete_stocks_without_prices(news_df,stock_list):

    symbols_of_df = news_df['symbol']
    in_or_not = [k in stock_list for k in symbols_of_df]
    
    return news_df[in_or_not]


#------------------------------------------
def topic_for_news (new_df,topic_num):
    topic_columns = ['topic'+str(n+1) for n in range(topic_num)]
    most_possible_topic_list = []
    for i in range(len(new_df)):
        topic_data_list = list(new_df.iloc[i][topic_columns])
        max_data = np.array(topic_data_list).max()        
        most_possible_topic_minus_1 = topic_data_list.index(max_data)
        topic_data_list[most_possible_topic_minus_1] = -1
        second_max_data = np.array(topic_data_list).max()
        
        if max_data > second_max_data:
            most_possible_topic_list.append(most_possible_topic_minus_1 + 1)     
        else:
            most_possible_topic_list.append(-1) 
    new_df['most possible topic'] = pd.Series(most_possible_topic_list,index = new_df.index)
    df = new_df[new_df['most possible topic'] > 0]
    return df


#------------------------------------------
def topic_count(news_df,topic_num,single_year):
    topic_columns = ['topic'+str(n+1) for n in range(topic_num)]
    topic_stock_df = pd.DataFrame()
    symbol_list = list(news_df['symbol'].unique())
    x=0
    for symbol in symbol_list:
        temp_df = news_df[news_df['symbol'] == symbol]
        week_list = list(temp_df['week_num'].unique())
        for week_num in week_list:
            
            temp_single_week_df = temp_df[temp_df['week_num']==week_num]
            count_dict = {str(k):0 for k in np.arange(1,topic_num+1)}

            most_possible_topic_series = temp_single_week_df['most possible topic']
            for most_possible_topic in most_possible_topic_series:
                count_dict[str(most_possible_topic)] += 1
            count_list = [(value,key) for key,value in count_dict.items()]
            count_list.sort(reverse=True)

            if count_list[0][0] > count_list[1][0]:
                x+=1
                topic_stock_df.loc[str(x),'symbol'] = symbol
                topic_stock_df.loc[str(x),'week_num'] = week_num
                topic_stock_df.loc[str(x),'single_year'] = single_year
                topic_stock_df.loc[str(x),'most possible topic'] = int(count_list[0][1])

                for k in range(topic_num):
                    topic_n = k+1
                    topic_column_name = topic_columns[k]
                    topic_stock_df.loc[str(x),'topic_count_'+str(topic_n)] = count_dict[str(topic_n)]
    return topic_stock_df
      
#------------------------------------------

def map_returns (topic_stock_df,filepath_csv,single_year = 2017,length_period=10):
    symbol_list = list(topic_stock_df['symbol'].unique())
    combine_df = pd.DataFrame()

    for symbol in symbol_list:
        temp_df = topic_stock_df[topic_stock_df['symbol'] == symbol]
        price_df = pd.read_csv(filepath_csv+symbol+'.csv')
        price_df = price_df[price_df['year']==single_year]
        single_combine_df = pd.merge(temp_df,price_df,how='inner',on='week_num')
        combine_df = pd.concat([combine_df,single_combine_df],axis=0,ignore_index=True)

    return combine_df

#------------------------------------------
def performance_single_topic(single_topic_df,length_period):
    performance_week_df = pd.DataFrame()
    all_weeks = pd.Series([i for i in np.arange(1,53)])
    week_list = list(single_topic_df['week_num'].unique())
    return_columns = ['simple_return_'+str(n+1) for n in range(length_period)]

    x = 0
    for week_num in all_weeks:
        x+=1
        if week_num in week_list:
            temp_df = single_topic_df[single_topic_df['week_num']==week_num]
            performance_week_df.loc[str(x),'week_num'] = week_num
            performance_week_df.loc[str(x),'have data'] = 'yes'
            
            for col in return_columns:
                performance_week_df.loc[str(x),col] = temp_df[col].mean()
        else:
            performance_week_df.loc[str(x),'week_num'] = week_num
            performance_week_df.loc[str(x),'have data'] = 'no'
            for col in return_columns:
                performance_week_df.loc[str(x),col] = 0        
            

    
    return performance_week_df.fillna(0)


#------------------------------------------
def performance_for_all_topics(combine_df,topic_num,length_period,filepath_csv_performance_output):
    for topic_n in np.arange(1,topic_num+1):
        single_topic_df = combine_df[combine_df['most possible topic'] == topic_n]
        performance_week_df =  performance_single_topic(single_topic_df,length_period)

        filename = filepath_csv_performance_output+'topic_'+str(topic_n)+'_performance.csv'
        performance_week_df.to_csv(filename)


def main():
    data_file = '/data.csv'
    filepath_csv = '/stockprices/'
    filepath_csv_performance_output = '/performance/'
    
    news_df = pd.read_csv(datafile)
    stock_list = collectFilename(filepath_csv,'csv')
    df = delete_stocks_without_prices(news_df,stock_list)
    df = week_number(df)
    df = topic_for_news(df,20)
    topic_stock_df = topic_count(df,20,2017)
    combine_df = map_returns (topic_stock_df,filepath_csv)
    performance_for_all_topics(combine_df,20,10,filepath_csv_performance_output)


if __name__ == '__main__':
    main()
