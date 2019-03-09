import pandas as pd
import pandas_datareader.data as web
from datetime import datetime, timedelta, date
from pandas_datareader._utils import RemoteDataError
import os
import fix_yahoo_finance as yf
import socket
import urllib
import json
import time
import nltk
import string
from nltk.corpus import stopwords
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from general_function import *


########################################
'''
Please change parameters and paths of input files here.
'''
prior = 2017  # In-sample test year
post = 2018   # Out-of-sample test year

# Please change the filepath for saving data here.
# Notice: Please ensure that the base_dir exists in your computer.
# Besides, please ensure the base_dir ends with '/'.
base_dir = 'D:/data/'   
check_and_create_dir(base_dir)

# Number of cumulative abnormal returns.
car_num = 10  

# Please add path of csv which contains 'Symbol' data here.
symbol_csv = 'D:/data/input/bbg_ticker_30-5-18.csv'   

# Change link every time when api is used. 
# api_link is used in "step 3: downloading news from internal API".
api_link = 'http://148.251.47.168:8080/api/v2/get_articles?class_nameid=c_'    
time_zone = None   

# Please change the filepath of stop words here. 
# The stop_words_txt should contain the additional stop words except for stop words from nltk.corpus package.
# This file is made used of in "step 5: text preprocessing".
stop_words_txt = 'D:/data/input/stop_words.txt'



#----------------------------------------------------------------
# Build dirs before downloading and running data.


common_dir_dict = {'stock_prices_dir': 'stock_prices/',
                   'dir_prices_yahoo': 'stock_prices/yahoo/',
                   'dir_prices_iex': 'stock_prices/iex/',
                   'dir_prices_index': 'input/index/',
                   'car_dir': 'stock_prices/car_{}'.format(car_num),
                   'strategy_data_dir': 'strategy_data/{}—{}/'.format(prior,post),
                   'performance_dir': 'performance/',
                   'input_dir': 'input/',
                   'common_news_dir': 'news/',
                   'lda_dir': 'new/lda_data/'}
            
in_sample_dir_dict = {'news_dir': 'news/news_{}'.format(prior),
                      'samples_news_dir': 'news/news_{}_{}_car/'.format(prior,car_num)]

out_of_sample_dir_dict = {'news_dir': 'news/news_{}'.format(post),
                          'samples_news_dir': 'news/news_{}_{}_car/'.format(post,car_num)}

for dir_dict in [common_dir_dict, in_sample_dir_dict, out_of_sample_dir_dict]:
    for name in dir_dict:
        temp_path = dir_dict[name]
        path = base_dir + temp_path
        check_and_create_dir(path)
        dir_dict[name] = path

strategy_dir = common_dir_dict['strategy_data_dir']
temp_1 = base_dir + strategy_dir + '{}_in_sample'.format(prior)
temp_2 = base_dir + strategy_dir + '{}_in_sample'.format(post)
check_and_create_dir(temp_1)
check_and_create_dir(temp_2)
in_sample_dir_dict['strategy_data_dir'] = temp_1
out_of_sample_dir_dict['strategy_data_dir'] = temp_2




#----------------------------------------------------------------
# Step 1: Download stock prices from Yahoo! Finance and IEX databases
from prices_download import download_all_prices             #  import function from prices_download.py
start_date = datetime(2000,1,1)
end_date = datetime.today()

dir_prices_1 = common_dir_dict['dir_prices_yahoo']
dir_prices_2 = common_dir_dict['dir_prices_iex']

f = symbol_csv
ticker_list = list(pd.read_csv(f)['Symbol'].unique())
download_all_prices(ticker_list, dir_prices_1, dir_prices_2, start_date, end_date)

# download index data
dir_prices_index = common_dir_dict['dir_prices_index']
download_all_prices(['SPY'], dir_prices_index, dir_prices_index, start_date, end_date)


#----------------------------------------------------------------
# Step 2: Get samples with abnormal returns
from abnormal_returns import outputResults

index_filename = dir_prices_index + 'SPY.csv'
index_df = pd.read_data(index_filename)
car_dir = common_dir_dict['car_dir']
L = 240
W = 10

t1 = collectStockName(dir_prices_1)
iex = False
outputResults(t1,dir_prices_1,index_df,car_dir,L,W,iex)

t2 = collectStockName(dir_prices_2)
iex = True
outputResults(t2,dir_prices_2,index_df,car_dir,L,W,iex)

from high_car import carSelect
output = carSelect(car_dir,car_column='CAR[-W:0]',t_column='CAR[-W:0] t-Score',car_criteria=0.1,t_criteria=1.96,days=11)
car_csv = common_dir_dict['input_dir'] + '{}_car.csv'.format(car_num)
output.to_csv(car_csv)



#----------------------------------------------------------------
# Step 3: Download annual news for all symbols, including samples
from news_download import news_function            # import news_function from news_download.py

a = collectStockName(dir_prices_1)
a2 = collectStockName(dir_prices_2)
a.extend(a2)

b = pd.read_csv(symbol_csv)        # path of symbol_csv is defined in the most beginning part.
symbol_slice = [i in a for i in b['Symbol']]
data_df = b[symbol_slice]
print('Total number of symbols needed to be downloaded: {}'.format(len(data_df)))


####
# Download all news for the "prior" one year.
year = prior
news_dir = in_sample_dir_dict['news_dir']
news_function(news_dir, data_df, year, api_link, time_zone)

####
# Download all news for the "post" one year.
year = post
news_dir = out_of_sample_dir_dict['news_dir']
news_function(news_dir, data_df, year, api_link, time_zone)



#----------------------------------------------------------------
# Step 4: Download news during event windows for only abnormal samples
from news_download import news_for_abnormal_samples

# Calculate and get samples with 10-day cumulative abnormal returns (CAR) higher than 10%.
car_df = pd.read_csv(car_csv)
symbol_df = pd.read_csv(symbol_csv)
car_df = pd.merge(car_df,symbol_df,how='inner',on='Symbol')

car_df['Date'] = pd.to_datetime(car_df['Date'])
delta = timedelta(days=W)
car_df['Start_Date'] = car_df['Date'] - delta
car_df['End_Date'] = car_df['Date']
car_df['year'] = car_df['Date'].dt.year
# Save data to car_csv
car_df.to_csv(car_csv)


# Download news of CAR samples that cover the prior one year.
### 
news_dir = in_sample_dir_dict['news_dir']
symbols = collectFilenames(news_dir,'json')
slices = [i in symbols for i in car_df['Symbol']]
temp_df = car_df[slices]

car_news_dir = common_dir_dict['common_news_dir'] + 'news_{}_{}_car/'.format(prior,car_num)
check_and_creat_dir(car_news_dir)
year = 2015
news_for_abnormal_samples(car_news_dir, temp_df, year, api_link, time_zone)


#----------------------------------------------------------------
# Step 5: News Preprocessing: news of samples
from news_preprocess import preprocessNewsViaPickle,saveAsPickle

symbol_series = pd.read_csv(symbol_csv)['Symbol']
symbol_list = [symbol.lower() for symbol in symbol_series ]

############
# Save all raw news in "prior" year and "post" year into two pickles.
# Pickles are for following analysis.
common_news_dir = common_dir_dict['common_news_dir']

news_dir_prior = in_sample_dir_dict['news_dir']
pickle_raw_news_prior = common_news_dir + '{}_raw_news.pickle'.format(prior)

news_dir_post = out_of_sample_dir_dict['news_dir']
pickle_raw_news_post = common_news_dir + '{}_raw_news.pickle'.format(post)

# Save all raw news (before preprocessing) into pickles.
saveAsPickle(news_dir_prior, pickle_raw_news_prior)
saveAsPickle(news_dir_post, pickle_raw_news_post)

# Preprocess raw news and save them into pickles.
# This step is for forming trading strategy, in-sample test and out-of-sample test
pickle_preprocess_news_prior = common_news_dir + '{}_preprocess_news.pickle'.format(prior)
pickle_preprocess_news_post = common_news_dir + '{}_preprocess_news.pickle'.format(post)

preprocessNewsViaPickle(pickle_raw_news_prior,
                        pickle_preprocess_news_prior,
                        stop_words_txt,
                        symbol_list)
preprocessNewsViaPickle(pickle_raw_news_post,
                        pickle_preprocess_news_post,
                        stop_words_txt,
                        symbol_list)

#############
# Save raw news (before preprocessing) for only abnormal samples into one pickle.
pickle_car = common_news_dir + '{}_news_for_{}_car_samples'.format(prior,car_num)
car_news_dir = common_dir_dict['common_news_dir'] + 'news_{}_{}_car/'.format(prior,car_num)
saveAsPickle(car_news_dir,pickle_car)

# Preprocess news for abnormal samples and save them into another pickle.
pickle_preprocess_news_car = common_news_dir + '{}_preprocess_news_for_{}_car.pickle'.format(prior,car_num)

preprocessNewsViaPickle(pickle_car,
                        pickle_preprocess_news_car,
                        stop_words_txt,
                        symbol_list)


#----------------------------------------------------------------
# Step 6: LDA Analysis
from lda_optimization import combineNewsViaPickle,ldaOptimization
'''
Run LDA algorithm based on preprocessed news of abnormal samples.
In this step, results for various topic numbers (20,25,30,35,40) would be all collected.
There would be a model complexity for each result.
'''
# lda_dir is for collecting results from LDA algorithm.
lda_dir = common_dir_dict['lda_dir'] + '{}-{}/'.format(prior,post)
check_and_create_dir(lda_dir)

# filename_output_txt is to collect top words from topics.
# This file only include top words for results with lowest model complexity.
filename_output_txt =  lda_dir + '{}_car_{}_top_words.txt'.format(car_num,year)
# filename_output_csv is to collect details, including time cost and model complexity, for all results.
filename_output_csv =  lda_dir + '{}_car_{}_lda_results.csv'.format(car_num,year)
# pickle_all_records is to collect all top words for all topic numbers.
pickle_all_records = lda_dir + '{}_car_{}_top_words_all_topics.pickle'.format(car_num,year)
# pickle_all_lda is to collect all data from LDA classes, covering all topic numbers.
# Hence, the LDA classes can be reused in next step,
pickle_all_lda = lda_dir + '{}_car_{}_lda_results.pickle'.format(car_num,year)

# pickle_preprocess_news_car has been defined at step 5.
# Collect preprocessed news for abnormal samples, and combine them into a single string list.
docLst = combineNewsViaPickle(pickle_preprocess_news_car)
topic_best, model_features = ldaOptimization(docLst,
                                             filename_output_txt,
                                             filename_output_csv,
                                             n_topics_list,
                                             fixed_top_words=200,
                                             fixed_iteration =2000,
                                             fixed_features=len(docLst),
                                             pickle_all_records = pickle_all_records,
                                             pickle_all_lda = pickle_all_lda)

#----------------------------------------------------------------
# Step 7: Map topics
'''
In last step, the LDA classes and results are collected.
The LDA results are only for samples with high abnormal returns.
In this step, we need to map all news during a year to the LDA class.
We will get the correlation with topics for each piece of news.
'''

from map_topics import map_topics_function

# In sample
load_lda_time = int((topic_best - 20) / 5 + 1))

mapping_main_function(lda_class_pickle=pickle_all_lda,
                      pickle_with_process_news=pickle_preprocess_news_prior,
                      n_topic=topic_best,
                      car_num=car_num,
                      load_lda_j=load_lda_time,
                      fix_features=model_features,
                      output_dir=strategy_dir,
                      single_year=prior)
# Out of sample
mapping_main_function(lda_class_pickle=pickle_all_lda,
                      pickle_with_process_news=pickle_preprocess_news_post,
                      n_topic=topic_best,
                      car_num=car_num,
                      load_lda_j=load_lda_time,
                      fix_features=model_features,
                      output_dir=strategy_dir,
                      single_year=post)


#----------------------------------------------------------------
# Step 8: Compute stock returns
from stock_returns import multi_simple_returns

'''
This part is for calculating weekly simple returns.
'''

returns_dir_prior = common_dir_dict['stock_prices_dir'] + '/yahoo_iex_{}/'.format(prior)
returns_dir_post = common_dir_dict['stock_prices_dir'] + '/yahoo_iex_{}/'.format(post)
check_and_create_dir(returns_dir_prior)
check_and_create_dir(returns_dir_post)

multi_simple_returns(dir_prices_1,
                     returns_dir_prior,
                     freq='W',
                     start_year = prior,
                     get_week_num = True,
                     iex = False)
multi_simple_returns(dir_prices_2,
                     returns_dir_prior,
                     freq='W',
                     start_year = prior,
                     get_week_num = True,
                     iex = True)
multi_simple_returns(dir_prices_1,
                     returns_dir_post,
                     freq='W',
                     start_year = post,
                     get_week_num = True,
                     iex = False)
multi_simple_returns(dir_prices_2,
                     returns_dir_post,
                     freq='W',
                     start_year = post,
                     get_week_num = True,
                     iex = True)


dir_prices_index = common_dir_dict['dir_prices_index']

returns_index_prior = dir_prices_index + 'returns_{}/'.format(prior)
returns_index_post = dir_prices_index + 'returns_{}/'.format(post)
check_and_create_dir(returns_index_prior)
check_and_create_dir(returns_index_post)

multi_simple_returns(dir_prices_index,
                     returns_index_prior,
                     freq='W',
                     start_year = prior,
                     get_week_num = True,
                     iex = False)

multi_simple_returns(dir_prices_index,
                     returns_index_post,
                     freq='W',
                     start_year = post,
                     get_week_num = True,
                     iex = False)

#----------------------------------------------------------------
# Step 8: Strategy
from strategy import main_function, main_pair_plot_function


#########################
topic_num = topic_best

# In-sample test
single_year = prior
temp_filepath =  in_sample_dir_dict['strategy_data_dir']


dir_prices_index = common_dir_dict['dir_prices_index']
returns_dir = common_dir_dict['stock_prices_dir'] + '/yahoo_iex_{}/'.format(single_year)
path_1 = temp_filepath + '{}_car_{}_topics/'.format(car_num,topic_num)
path_2 = path_1 + 'strategy_performance/'
path_3 = path_2 + 'performance_of_each_topic/'
performance_dir_prior_1 = path_1
performance_dir_prior_2 = path_2
performance_dir_prior_3 = path_3

for path in [path_1,path_2,path_3]:
    check_and_create_dir(path)

set_end_week = None
length_period = 1
topic_size = 5
benchmark_filename = dir_prices_index + 'returns_{}/SPY.csv'.format(single_year)
dates_list = pd.read_csv(benchmark_filename)['Date']
dates_list = list(dates_list)

car_topic_pairs = [(car_num,topic_num)]
main_function(car_topic_pairs=car_topic_pairs,
              single_year=single_year,
              filepath_csv=returns_dir,
              temp_filepath=temp_filepath, 
              benchmark_filename=benchmark_filename,
              dates_list=dates_list,
              topic_size=topic_size,
              set_end_week=set_end_week,
              length_period=length_period)

#########################
# Out-of-sample test
single_year = post
temp_filepath =  out_of_sample_dir_dict['strategy_data_dir']


dir_prices_index = common_dir_dict['dir_prices_index']
returns_dir = common_dir_dict['stock_prices_dir'] + '/yahoo_iex_{}/'.format(single_year)
path_1 = temp_filepath + '{}_car_{}_topics/'.format(car_num,topic_num)
path_2 = path_1 + 'strategy_performance/'
path_3 = path_2 + 'performance_of_each_topic/'
performance_dir_post_1 = path_1
performance_dir_post_2 = path_2
performance_dir_post_3 = path_3

for path in [path_1,path_2,path_3]:
    check_and_create_dir(path)

set_end_week = None
length_period = 1
topic_size = 5
benchmark_filename = dir_prices_index + 'returns_{}/SPY.csv'.format(single_year)
dates_list = pd.read_csv(benchmark_filename)['Date']
dates_list = list(dates_list)

car_topic_pairs = [(car_num,topic_num)]
main_function(car_topic_pairs=car_topic_pairs,
              single_year=single_year,
              filepath_csv=returns_dir,
              temp_filepath=temp_filepath, 
              benchmark_filename=benchmark_filename,
              dates_list=dates_list,
              topic_size=topic_size,
              set_end_week=set_end_week,
              length_period=length_period)



#----------------------------------------------------------------
# Step 9: Performance
benchmark_filename = dir_prices_index + 'returns_{}/SPY.csv'.format(post)
benchmark_df = pd.read_csv(benchmark_df)
benchmark_df = benchmark_df.set_index('week_num')
benchmark_profit = benchmark_df['next_week_return']
car_topic_pairs = [(car_num,topic_num)]

factor_list = ['Jensen Alpha','Annualized Returns','Sharpe Ratio','Informance_ratio']
output_root_dir = common_dir_dict['performance_dir']
for a in factor_list:
    path = output_root_dir + factor_list + '/'
    check_and_create_dir(path)

factor_dict= {'Jensen Alpha':[10,0.35,0.3,0.25,0.2,0.15,0.1],
              'Annualized Returns':[10,0.5,0.45,0.4,0.35,0.3,0.25,0.2],
              'Sharpe Ratio':[10,2,1.5,1,0.5],
              'Information Ratio':[10,2,1.5,1,0.5]}

end_week = None
group_size = 5
root_dir_1 = performance_dir_prior_1
root_dir_2 = performance_dir_post_1

k = 0
for car_num,topic_num in car_topic_pairs:
    for factor, filter_list in factor_dict.items():
        k+=1
        print(k)
        group_performance_function(car_num,topic_num,group_size,factor,filter_list,root_dir_1,root_dir_2,output_root_dir,benchmark_profit,end_week)
        print('***DONE***')




