import os
import pandas as pd
import numpy as np
import datetime
import statsmodels.api as sm
import matplotlib.pyplot as plt

#------------------------------------------ 
def collectFilename(filepath,filetype='csv'):
    filename_list=[]
    for root, dirs, files in os.walk(filepath):
        if files:
            for f in files:
                if filetype in f:
                    filename_list.append(f.split('.'+filetype)[0])
    return filename_list

#------------------------------------------        
def week_number(df):
    '''
    Get the number of week in a specific year, according to "Date".

    Args:
    df: DataFrame, should include a column 'Date'

    '''
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

#------------------------------------------
def delete_stocks_without_prices(news_df,stock_list):
    '''
    If there is no price available price data for a specific news, delete the news.
    Args:
    news_df: DataFrame, contains news data
    stock_list: list, stocks with available prices data 
    '''
    symbols_of_df = news_df['symbol']
    in_or_not = [k in stock_list for k in symbols_of_df]
    return news_df[in_or_not]

#------------------------------------------
def topic_for_news (new_df,topic_num):
    '''
    Map news to topics, according to largest correlation data.

    Args:
    new_df(pd.DataFrame): df with correlation data
    topic_num(int): number of topics
    
    '''
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
    '''
    This is to count how many news are mapped to topics for a specific stock in a week.
    Args:
    news_df(pd.DataFrame): df that contains correlation data and most possible topic for each news
    topic_num(int): number of topics
    single_year(int): year
    
    '''

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
                    topic_stock_df.loc[str(x),'topic_count_'+str(topic_n)] = count_dict[str(topic_n)]
    return topic_stock_df      
#------------------------------------------

def fix_topic_size(topic_stock_df,topic_num,topic_size):
    '''
    For a topic in a week, if there are more than 5 stocks mapped to same topic,
    choose stocks with largest number of news mapped to that topic.
    If there are less than 5 stocks mapped to same topic in a week, 
    the stocks are all included and taken into consideration in that week.

    Args:
    topic_stock_df(pd.DataFrame): df that contains topic counts data
    topic_num(int): number of topics
    topic_size(int): equals to five in this research
    '''
    output = pd.DataFrame()
    topic_count_columns = ['topic_count_'+str(i+1) for i in range(topic_num)]
    week_list = list(topic_stock_df['week_num'].unique())
    
    for week_num in week_list:
        single_week  = topic_stock_df[topic_stock_df['week_num'] == week_num]
    
        for i in range(topic_num):
            topic_n = i + 1
            
            try:
                temp_df = single_week[single_week['most possible topic'] == topic_n]
            except:
                pass
            else:
                topic_c = topic_count_columns[i]           
                
                if len(temp_df) > 0:
                    if len(temp_df) > topic_size:
                        temp_df = temp_df.sort_values(by = topic_c, ascending = False)
                        temp_df = temp_df.iloc[:topic_size]
                    
                    output = pd.concat([output,temp_df],ignore_index = True)
    return output
        
#------------------------------------------
def map_returns (topic_stock_df,filepath_csv,single_year):
    '''
    Combine stocks and weekly returns.
    Args:
    topic_stock_df: df contains stocks data
    filepath_csv(str): filepath that includes returns csv
    single_year(int): year
    '''
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
def performance_single_topic(single_topic_df,
                             dates_list,
                             end_week):  
    '''
    Get weekly performances for a single topic.
    Args:
    single_topic_df(pd.DataFrame): df that contains data for a single topic
    dates_list(list): a list of dates
    end_week(int): if data covers a whole year, this would be 52. else: this would be the latest week number in a year
    
    '''  
    performance_week_df = pd.DataFrame()
    if end_week is None:
        all_weeks = pd.Series([i for i in np.arange(1,53)])
    else:
        all_weeks = pd.Series([i for i in np.arange(1,end_week+1)])
        dates_list = dates_list[:end_week]        
    week_list = list(single_topic_df['week_num'].unique())
    return_columns = ['next_week_return']
    x = 0
    for week_num in all_weeks:
        x+=1
        if week_num in week_list:
            temp_df = single_topic_df[single_topic_df['week_num']==week_num]
            performance_week_df.loc[str(x),'week_num'] = week_num
            performance_week_df.loc[str(x),'have data'] = 'yes'
            performance_week_df.loc[str(x),'number of stocks'] = len(temp_df)
            symbol_list = list(temp_df['symbol'])
            symbol_sum = ''
            for symbol in symbol_list:
                if symbol_sum == '':
                    symbol_sum = symbol_sum + symbol
                else:
                    symbol_sum = symbol_sum+','+symbol
            performance_week_df.loc[str(x),'symbols'] = symbol_sum

            # equally weighted
            for col in return_columns:
                performance_week_df.loc[str(x),col] = temp_df[col].mean()
        else:
            performance_week_df.loc[str(x),'week_num'] = week_num
            performance_week_df.loc[str(x),'have data'] = 'no'
            performance_week_df.loc[str(x),'number of stocks'] = 0
            performance_week_df.loc[str(x),'symbols'] = 'No symbol'
            
            for col in return_columns:
                performance_week_df.loc[str(x),col] = 0      

    if dates_list is not None:
        performance_week_df['Date']=pd.Series(dates_list,index=performance_week_df.index)
    return performance_week_df.fillna(0)

#------------------------------------------
def performance_for_all_topics(combine_df,
                               topic_num,
                               filepath_csv_performance_output,
                               dates_list,
                               end_week):
    '''
    Output performances for all topics.
    combine_df(pd.DataFrame): df that contains topic data and return data
    topic_num(int): number of topics
    filepath_csv_performance_output(str): filepath for saving performances csv
    dates_list(list): a list of dates
    end_week(int): if data covers a whole year, this would be 52. else: this would be the latest week number in a year
    '''
    for topic_n in np.arange(1,topic_num+1):
        single_topic_df = combine_df[combine_df['most possible topic'] == topic_n]
        performance_week_df =  performance_single_topic(single_topic_df,
                                                        dates_list,
                                                        end_week)
        filename = filepath_csv_performance_output+'topic_'+str(topic_n)+'_performance.csv'
        performance_week_df.to_csv(filename)


#----------------------------------------
def remove_low_relevance(df):
    df = df[df['relevance'] != 'low']
    return df

def remove_low_sentiment(df):
    df = df[df['sentiment'] > 5]
    return df
#----------------------------------------
def main_strategy_function(car_num,
                           single_year,
                           news_df_filename,
                           filepath_csv,
                           temp_filepath,
                           topic_num,
                           topic_size = 5,
                           dates_list=None,
                           end_week=None):
    '''
    Combine all functions above for outputing results and can be directly used.
    Args:
    car_num(int): number of car
    single_year(int): year
    news_df_filename(str): filename of most original csv that contains correlation data of news
    filepath_csv(str): filepath for stock returns
    temp_filepath(str): root dir for saving results
    topic_num(int): number of topics
    topic_size(int): number for controlling maximum number of stocks in same topic
    dates_list(list): a list of topics
    end_week(int): if data covers a whole year, this would be 52. else: this would be the latest week number in a year

    '''
    news_df = pd.read_csv(news_df_filename)
    stock_list = collectFilename(filepath_csv)
    output_filepath = temp_filepath+str(car_num)+'_car_'+str(topic_num)+'_topics/'
    df = delete_stocks_without_prices(news_df,stock_list)
    df = week_number(df)

    df.to_csv(output_filepath+'strategy_performance/1_news_only_stocks_with_prices.csv')
    df = topic_for_news(df,topic_num)
    df.to_csv(output_filepath+'strategy_performance/2_news_possible_topics.csv')
    topic_stock_df = topic_count(df,topic_num,single_year)
    topic_stock_df = fix_topic_size(topic_stock_df,topic_num,topic_size)
    topic_stock_df.to_csv(output_filepath+'strategy_performance/3_stocks_possible_topics.csv')
    combine_df = map_returns (topic_stock_df,filepath_csv,single_year)
    combine_df.to_csv(output_filepath+'strategy_performance/4_stocks_combine_with_returns.csv')
    filepath_csv_performance_output = output_filepath+'strategy_performance/performance_of_each_topic/'
    performance_for_all_topics(combine_df,topic_num,filepath_csv_performance_output,dates_list,end_week)
 

#---------------------------------------
# Functions below are for computing strategy performances.
     
def Sum_profit(week_profit):
    sum_profit = week_profit.cumsum()
    return sum_profit
#---------------------------------------
def Annual_profit(week_num, sum_profit):
    data = {'week_num': week_num,'sum_profit': sum_profit}
    dataframe = pd.DataFrame(data)
    trade_weeks = len(dataframe.index)
    annual_profit = dataframe.sum_profit.iloc[-1]*52/trade_weeks
    return annual_profit
#---------------------------------------
def Max_drawdown(week_num, sum_profit):   
    data = {'week_num': week_num,'sum_profit': sum_profit}
    dataframe = pd.DataFrame(data)
    dataframe['max2here'] = dataframe['sum_profit'].cummax()
    dataframe['drawdown'] = dataframe['sum_profit'] - dataframe['max2here']
    temp = dataframe.sort_values(by = 'drawdown').iloc[0]
    max_drawdown = temp.drawdown
    return max_drawdown

#---------------------------------------
def Week_win_chance(week_num,week_profit):
    data = {'week_num':week_num,'week_profit':week_profit}
    dataframe = pd.DataFrame(data)
    week_win_chance = len(dataframe[dataframe['week_profit'] > 0 ])/len(dataframe)
    return week_win_chance
#---------------------------------------
def Max_sequent_weeks(week_num,week_profit):
    data = {'week_num':week_num,'week_profit':week_profit}
    dataframe = pd.DataFrame(data)
    if dataframe.week_profit[0] > 0:
        dataframe.loc[0, 'count'] = 1
    else:
        dataframe.loc[0, 'count'] = -1
    for i in dataframe.index[1:]:
        if dataframe.week_profit[i] > 0 and dataframe.week_profit[i - 1] > 0:
            dataframe.loc[i, 'count'] = dataframe.loc[i-1,'count'] + 1
        elif dataframe.week_profit[i] <= 0 and dataframe.week_profit[i - 1] <= 0:
            dataframe.loc[i, 'count'] = dataframe.loc[i-1,'count']-1
        elif dataframe.week_profit[i] > 0 and dataframe.week_profit[i - 1] <= 0:
            dataframe.loc[i, 'count'] = 1
        elif dataframe.week_profit[i] <= 0 and dataframe.week_profit[i - 1] > 0:
            dataframe.loc[i, 'count'] = -1
    dataframe.count = list(dataframe['count'])
    return max(dataframe.count), min(dataframe.count)
#---------------------------------------
def VIX(week_profit):
    return np.std(week_profit)*np.sqrt(52)
#---------------------------------------
def Sharp_ratio(annual_profit, VIX):
    return (annual_profit - 0.025)/VIX
#---------------------------------------
def Infromation_ratio(week_profit, benchmark_profit):
    diff = pd.Series(week_profit - benchmark_profit)
    return diff.mean() * 52/(diff.std() * np.sqrt(52))
#---------------------------------------
def to_percent(temp, position):
    return '%1.00f'%(100*temp) + '%'
#---------------------------------------
from matplotlib.ticker import FuncFormatter
def main_plot_function(benchmark_filename,
                       car_num,topic_num,
                       single_year,
                       length_period,
                       news_df_filename,
                       filepath_csv,
                       temp_filepath,
                       topic_i = 1,
                       topic_j = 10,
                       rows_plot = 5,
                       end_week=None):
    
    output_filepath = temp_filepath+str(car_num)+'_car_'+str(topic_num)+'_topics/'
    filepath_csv_performance_output = output_filepath+'strategy_performance/performance_of_each_topic/'

    
    benchmark_df = pd.read_csv(benchmark_filename)
    benchmark_df = benchmark_df[benchmark_df['year']==single_year]
    
    if end_week is not None:
        benchmark_df = benchmark_df[benchmark_df['week_num']<=end_week]
    benchmark_df['Date'] =pd.to_datetime(benchmark_df['Date'])
    benchmark_df = benchmark_df.set_index('Date').sort_index() 
    benchmark_profit = benchmark_df['next_week_return'].shift(1).fillna(0)
    
    plt.figure(figsize=(20,25))
    x = 0
    for topic_n in np.arange(topic_i,topic_j):
        x+=1       
        filename = 'topic_'+str(topic_n)+'_performance'
        news_df = pd.read_csv(filepath_csv_performance_output+filename+'.csv')
        if end_week is not None:
            news_df = news_df[news_df['week_num']<=end_week]
        news_df['Date'] = pd.to_datetime(news_df['Date'])
        news_df = news_df.set_index('Date')
        news_df = news_df.sort_index()
        idx = news_df.index
        week_profit = news_df['next_week_return'].shift(1).fillna(0)
                
        plt.subplot(rows_plot,2,x)
        plt.plot(idx,week_profit.cumsum(),'-',linewidth=3.0)     
        plt.plot(idx,benchmark_profit.cumsum(),'-.',linewidth=3.0)
        plt.legend(['Holding One Week','S&P 500'])
        plt.grid(True)
        plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percent))
        plt.title('Topic '+str(topic_n),fontsize=15)
        plt.ylabel('Cumulative Returns')
    fig = output_filepath+'strategy_performance/performance_'+str(topic_i)+'_to_'+str(topic_j)+'.png'
    plt.savefig(fig, format='png')
    print('PNG has been saved')

######################################
from statsmodels import regression
def main_performance_function(benchmark_filename,car_num,topic_num,single_year,length_period,news_df_filename,filepath_csv,temp_filepath,end_week=None):
    '''
    Get performances of topics.
    Args:
    benchmark_filename(str): name of csv file that contains S$P 500 returns
    car_num(int): number of car
    topic_num(int): number of topics
    single_year(int): year
    length_period(int): 1 in this research, because strategy is for holding one week
    news_df_filename(str): name of csv file that contains most original correlation data of news
    filepath_csv(str): filepath for stock returns
    temp_filepath(str): root dir for outputing results
    end_week(int): if data covers a whole year, this would be 52. else: this would be the latest week number in a year


    '''
    output_filepath = temp_filepath+str(car_num)+'_car_'+str(topic_num)+'_topics/'
    filepath_csv_performance_output = output_filepath+'strategy_performance/performance_of_each_topic/'
    benchmark_df = pd.read_csv(benchmark_filename)
    if end_week is None:
        k = 51
    else:
        k = int(end_week-1)

    benchmark_df = benchmark_df[benchmark_df['week_num']<=k]
    output = pd.DataFrame()
    
    x = 0

    for topic_n in np.arange(1,topic_num+1):
        filename = 'topic_'+str(topic_n)+'_performance'
        news_df = pd.read_csv(filepath_csv_performance_output+filename+'.csv')

        news_df = news_df[news_df['week_num']<=k]
            
        week_num = news_df['week_num']        
        
        ### RE
        average_stocks_num = news_df['number of stocks'].sum()/k
        weeks_with_signals = len(news_df[news_df['have data']=='yes'])

        x+=1
        week_profit = news_df['next_week_return']

      
        benchmark_profit = benchmark_df['next_week_return']

        sum_profit = Sum_profit(week_profit)
        annual_profit = Annual_profit(week_num, sum_profit)
 
        mdd = Max_drawdown(week_num, sum_profit)
        vix = VIX(week_profit)
        sr = Sharp_ratio(annual_profit, vix)
        ir = Infromation_ratio(week_profit, benchmark_profit)
        week_win_chance = Week_win_chance(week_num,week_profit)
        max_win_weeks, max_lose_weeks = Max_sequent_weeks(week_num,week_profit)

        output.loc[x,'Year'] = single_year
        output.loc[x,'Topic Number'] = int(topic_n)

        output.loc[x,'Annualized Returns'] = annual_profit
        output.loc[x,'Max Drawdown'] = mdd
        output.loc[x,'Sharpe Ratio'] = sr
        output.loc[x,'Information Ratio'] = ir
        
        X = np.array(benchmark_profit)
        Y = np.array(week_profit)
        X = X.reshape(-1,)

        X = sm.add_constant(X)
        MODEL= regression.linear_model.OLS(Y, X).fit()
        output.loc[x,'Treynor Ratio'] = annual_profit/MODEL.params[1]
        output.loc[x,'Jensen Alpha'] = annual_profit - MODEL.params[1]*(np.array(benchmark_profit).mean()*52) 
        
        output.loc[x,'Winning Chance'] = week_win_chance
        output.loc[x,'Max Sequent Winning Weeks'] = max_win_weeks
        output.loc[x,'Max Sequent Losing Weeks'] = -max_lose_weeks
        
        output.loc[x,'Average Number of Holding Stocks Per Week'] = average_stocks_num 
        output.loc[x,'Number of Trading Weeks'] = weeks_with_signals

    output.to_csv(output_filepath+'SummaryOfPerformance.csv')



################################
def main_function(car_topic_pairs,single_year,filepath_csv,temp_filepath, benchmark_filename,dates_list,topic_size,set_end_week,length_period):
    for i in range(len(car_topic_pairs)):
        car_num,topic_num = car_topic_pairs[i]
        
        print('--------  car  ---  '+str(car_num))
        print('########### topic ---  '+str(topic_num))
        news_df_filename = temp_filepath + str(car_num) +'_car_' + str(topic_num) + '_topics/' + str(single_year) + 'summary_all_news_' + str(car_num) + 'car_' +str(topic_num) + 'topics_full.csv'

        main_strategy_function(car_num,
                                single_year,
                                news_df_filename,
                                filepath_csv,
                                temp_filepath,
                                topic_num,
                                topic_size = topic_size,
                                dates_list = dates_list,
                                end_week=set_end_week)
        

        
        for plot_time in range(topic_num//10):
            topic_i = plot_time *10 + 1
            topic_j = plot_time *10 + 11
            main_plot_function(benchmark_filename,car_num,topic_num,single_year,length_period,news_df_filename,filepath_csv,temp_filepath,
                                end_week=set_end_week,topic_i=topic_i,topic_j=topic_j,rows_plot=5)
        if topic_num % 10 > 0:
            rest = topic_num - topic_j + 1
            if rest % 2 == 0 :
                rows_plot = rest // 2
            else:
                rows_plot = rest // 2 + 1
            topic_i = topic_j
            topic_j = topic_num
            main_plot_function(benchmark_filename,car_num,topic_num,single_year,length_period,news_df_filename,filepath_csv,temp_filepath,
                                end_week=set_end_week,topic_i=topic_i,topic_j=topic_j,rows_plot=rows_plot)
            main_performance_function(benchmark_filename,car_num,topic_num,single_year,length_period,news_df_filename,filepath_csv,temp_filepath,end_week=set_end_week)
        
        main_performance_function(benchmark_filename, car_num, topic_num, single_year, length_period, news_df_filename, filepath_csv, temp_filepath, end_week=set_end_week)
            
            
        print('Done----------------')











###############################################3

def Main():    
    year_pairs_list = [(2015,2016),(2016,2017),(2017,2018)]
    car_topic_pairs_list = [[(10,20),(10,30),(20,20),(20,40)],[(10,20),(10,40),(20,20),(20,40)], [(10,20),(10,30),(20,20),(20,35)]]
    end_weeks_list = [(None,None),(None,None),(None,27)]
    
    for i in range(3):
        prior,post = year_pairs_list[i]
        car_topic_pairs = car_topic_pairs_list[i]
        end_week_pairs = end_weeks_list[i]
        # in-sample test
        
        single_year = prior
        
        base_dir = 'D:/data/strategy_data/{}-{}/'.format(prior, post)
        index_dir = 'D:/data/input/index/'
        filepath_csv = 'D:/data/StockPrices/yahoo_iex_{}/'.format(single_year)
        
    
        temp_filepath =  base_dir +'{}_in_sample/'.format(single_year)
        
        set_end_week = end_week_pairs[0]
        length_period = 1
        topic_size = 5
        benchmark_filename = index_dir + 'returns_{}/SPY.csv'.format(single_year)
        dates_list = pd.read_csv(benchmark_filename)['Date']
        dates_list = list(dates_list)
        
        main_function(car_topic_pairs,single_year,filepath_csv,temp_filepath, benchmark_filename,dates_list,topic_size,set_end_week,length_period)
        
        #########################
        # output-of-sample test
        single_year = post
        
        base_dir = 'D:/data/strategy_data/{}-{}/'.format(prior, post)
        index_dir = 'D:/data/input/index/'
        filepath_csv = 'D:/data/StockPrices/yahoo_iex_{}/'.format(single_year)
        temp_filepath =  base_dir +'{}_out_of_sample/'.format(single_year)
        
        set_end_week = end_week_pairs[1]
        length_period = 1
        topic_size = 5
        benchmark_filename = index_dir + 'returns_{}/SPY.csv'.format(single_year)
        dates_list = pd.read_csv(benchmark_filename)['Date']
        dates_list = list(dates_list)

        main_function(car_topic_pairs,single_year,filepath_csv,temp_filepath, benchmark_filename,dates_list,topic_size,set_end_week,length_period)
 
        
if __name__ == '__main__':
    Main()
    