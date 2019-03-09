from statsmodels import regression
import pandas as pd
import numpy as np
import statsmodels.api as sm



def performance_of_topic_group(topic_num,car_num,group_size, factor, filter_list, root_dir_1,root_dir_2,benchmark_profit,group_summay_path,end_week=None):
    '''
    Get performances of topics.
    topic_num(int): number of topics
    car_num(int): number of car
    group_size(int): 5 in this research
    factor(str): factor for selecting best 5 topics
    filter_list(list): can be ignored.
    root_dir_1(str): dir for in-sample data
    root_dor_2(str): dir for out-of-sample data
    benchmark_profit(pd.Series): returns of market index
    group_summay_path(str): filepath for outputing summary results
    end_week(int): if data covers the whole year, end_week is 53. Else: end_week is latest week number in a year.

    '''
    insample_filepath = root_dir_1+str(car_num)+'_car_'+str(topic_num)+'_topics/'
    out_filepath = root_dir_2+str(car_num)+'_car_'+str(topic_num)+'_topics/'
    details_path = out_filepath+'strategy_performance/performance_of_each_topic/'
    
    output = pd.DataFrame()
    performance = pd.DataFrame()

    x = 0
    if end_week is None:
        end_week = 52
    else:
        end_week = end_week

    x += 1
    summary_df = pd.read_csv(insample_filepath+'SummaryOfPerformance.csv')
    summary_df = summary_df.sort_values(by=factor, ascending=False)
    topics = list(summary_df['Topic Number'])

    filter_start = round(topic_num/7)
    topics = topics[filter_start:]
    

    summary_df = summary_df.set_index('Topic Number')

    for i in range(group_size):
        topic_n = int(topics[i])
        temp_summary = summary_df.loc[topic_n,:]

        output.loc[x,'Filter'] = 'Delete 1/7 top topics'
        output.loc[x,'Topic No.'+str(i+1)] = topic_n
        output.loc[x,str(i+1)+' '+factor] = temp_summary[factor]

        filename = details_path+'topic_'+str(topic_n)+'_performance.csv'
        topic_n_df = pd.read_csv(filename)
        topic_n_df = topic_n_df[topic_n_df['week_num'] <  end_week]
        week_num = topic_n_df['week_num']
        topic_n_df = topic_n_df.set_index('week_num')
        profit = topic_n_df['next_week_return']
        
        
        if i == 0:
            week_profit = profit
        else:
            week_profit  = profit + week_profit

    week_profit = week_profit/group_size       
    
    sum_profit = Sum_profit(week_profit)
    annual_profit = Annual_profit(week_num, sum_profit)
    mdd = Max_drawdown(week_num, sum_profit)
    vix = VIX(week_profit)
    sr = Sharp_ratio(annual_profit, vix)
    ir = Infromation_ratio(week_profit, benchmark_profit)
    week_win_chance = Week_win_chance(week_num,week_profit)


    performance.loc[x,'Filter'] = 'Delete 1/7 top topics'
    performance.loc[x,'Annualized Returns'] = annual_profit
    performance.loc[x,'Max Drawdown'] = mdd
    performance.loc[x,'Sharpe Ratio'] = sr
    performance.loc[x,'Information Ratio'] = ir

    X = np.array(benchmark_profit)
    Y = np.array(week_profit)
    X = X.reshape(-1,)

    X = sm.add_constant(X)
    MODEL= regression.linear_model.OLS(Y, X).fit()
    performance.loc[x,'Treynor Ratio'] = annual_profit/MODEL.params[1]
    performance.loc[x,'Jensen Alpha'] = annual_profit - MODEL.params[1]*(np.array(benchmark_profit).mean()*52)
    performance.loc[x,'Winning Chance'] = week_win_chance


    #########################
    combine = pd.DataFrame()
    
    for filter in filter_list:
        x += 1

        summary_df = pd.read_csv(insample_filepath+'SummaryOfPerformance.csv')
        summary_df = summary_df[summary_df[factor] <= filter]
        summary_df = summary_df.sort_values(by=factor, ascending=False)
        topics = list(summary_df['Topic Number'])
        summary_df = summary_df.set_index('Topic Number')


        for i in range(group_size):
            topic_n = int(topics[i])
            temp_summary = summary_df.loc[topic_n,:]

            output.loc[x,'Filter'] = filter
            output.loc[x,'Topic No.'+str(i+1)] = topic_n
            output.loc[x,str(i+1)+' '+factor] = temp_summary[factor]

      
            filename = details_path+'topic_'+str(topic_n)+'_performance.csv'
            topic_n_df = pd.read_csv(filename)
            topic_n_df = topic_n_df[topic_n_df['week_num'] <  end_week]
            week_num = topic_n_df['week_num']
            topic_n_df = topic_n_df.set_index('week_num')
            profit = topic_n_df['next_week_return']
            
            if filter == 10:
                combine['topic_'+str(i+1)] = topic_n_df['next_week_return']
                combine['topic_'+str(i+1)+'_have_data'] = topic_n_df['have data']
                combine['topic_'+str(i+1)+'_stock_num'] = topic_n_df['number of stocks']
                combine['topic_'+str(i+1)+'_symbols'] = topic_n_df['symbols']
            

            if i == 0:
                week_profit = profit
            else:
                week_profit  = profit + week_profit


        week_profit = week_profit/group_size       
        
        sum_profit = Sum_profit(week_profit)
        annual_profit = Annual_profit(week_num, sum_profit)
        mdd = Max_drawdown(week_num, sum_profit)
        vix = VIX(week_profit)
        sr = Sharp_ratio(annual_profit, vix)
        ir = Infromation_ratio(week_profit, benchmark_profit)
        week_win_chance = Week_win_chance(week_num,week_profit)

        performance.loc[x,'Filter'] = filter
        performance.loc[x,'Annualized Returns'] = annual_profit
        performance.loc[x,'Max Drawdown'] = mdd
        performance.loc[x,'Sharpe Ratio'] = sr
        performance.loc[x,'Information Ratio'] = ir

        X = np.array(benchmark_profit)
        Y = np.array(week_profit)
        X = X.reshape(-1,)

        X = sm.add_constant(X)
        MODEL= regression.linear_model.OLS(Y, X).fit()
        performance.loc[x,'Treynor Ratio'] = annual_profit/MODEL.params[1]
        performance.loc[x,'Jensen Alpha'] = annual_profit - MODEL.params[1]*(np.array(benchmark_profit).mean()*52)
        performance.loc[x,'Winning Chance'] = week_win_chance
    
    return output,performance,combine




#############################

#----------------------------
def Sum_profit(week_profit):
    sum_profit = week_profit.cumsum()
    return sum_profit
    
#----------------------------
def Annual_profit(week_num, sum_profit):
    data = {'week_num': week_num,'sum_profit': sum_profit}
    dataframe = pd.DataFrame(data)
    trade_weeks = len(dataframe.index)
    annual_profit = dataframe.sum_profit.iloc[-1]*52/trade_weeks
    return annual_profit
#----------------------------
def Max_drawdown(week_num, sum_profit):
    
    data = {'week_num': week_num,'sum_profit': sum_profit}
    dataframe = pd.DataFrame(data)
    dataframe['max2here'] = dataframe['sum_profit'].cummax()
    dataframe['drawdown'] = dataframe['sum_profit'] - dataframe['max2here']
    temp = dataframe.sort_values(by = 'drawdown').iloc[0]
    max_drawdown = temp.drawdown

    return data['drawdown'], max_drawdown

#----------------------------
def Week_win_chance(week_num,week_profit):
    data = {'week_num':week_num,'week_profit':week_profit}
    dataframe = pd.DataFrame(data)
    week_win_chance = len(dataframe[dataframe['week_profit'] > 0 ])/len(dataframe)
    return week_win_chance
#----------------------------
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

#----------------------------
def VIX(week_profit):
    return np.std(week_profit)*np.sqrt(52)

#----------------------------
def Sharp_ratio(annual_profit, VIX):
    return (annual_profit - 0.025)/VIX
#----------------------------
def Infromation_ratio(week_profit, benchmark_profit):
    diff = pd.Series(week_profit - benchmark_profit)
    return diff.mean() * 52/(diff.std() * np.sqrt(52))




###############################
#----------------------------

def group_performance_function(car_num,topic_num,group_size,factor,filter_list,root_dir_1,root_dir_2,output_root_dir,benchmark_profit,end_week):
    
    filename = output_root_dir + '/' + factor +'/' + str(car_num) + '_' + str(topic_num) + factor + '.csv'
    filename_2 = output_root_dir + '/' + factor +'/' + str(car_num) + '_' + str(topic_num) + factor + '_2.csv'
    filename_3 = output_root_dir + '/' + factor +'/' + str(car_num) + '_' + str(topic_num) + factor + '_return_data.csv'
    
    group_summary_path = output_root_dir+ '/' + factor +'/' +str(car_num)+'_car_'+ str(topic_num) +'_topics/'
    output,performance,combine = performance_of_topic_group(topic_num = topic_num,
                                                            car_num = car_num,
                                                            group_size = group_size,
                                                            factor = factor,
                                                            filter_list = filter_list,
                                                            root_dir_1 = root_dir_1,
                                                            root_dir_2 = root_dir_2,
                                                            benchmark_profit = benchmark_profit,
                                                            group_summay_path = group_summary_path,
                                                            end_week = end_week)
    output.to_csv(filename)
    performance.to_csv(filename_2)
    combine.to_csv(filename_3)
    
    print('---Done---')





###############################################

def Main():

#=============================
    prior_post_pairs = [(2015,2016),(2016,2017),(2017,2018)]
    end_week_list = [53,53,28]
    car_topic_list = [[(10,20),(10,30),(20,20),(20,40)], [(10,20),(10,40),(20,20),(20,40)], [(10,20),(10,30),(20,20),(20,35)] ]
    
    for k in range(3):
        prior,post = prior_post_pairs[k]
        set_end_week = end_week_list[k]
        car_topic_pairs = car_topic_list[k]
        
        benchmark_df = pd.read_csv('D:/data/input/index/returns_{}/SPY.csv'.format(post))
    
        base_dir = 'D:/data/strategy_data/{}-{}/'.format(prior, post)
    
        root_dir_1 = base_dir + '{}_in_sample/'.format(prior)
        root_dir_2 = base_dir + '{}_out_of_sample/'.format(post)
    
        output_root_dir = 'D:/strategy_results/{}-{}'.format(prior, post)
    
        if set_end_week is not None:
            benchmark_df = benchmark_df[benchmark_df['week_num'] < set_end_week]
        benchmark_df = benchmark_df.set_index('week_num')
    
        benchmark_profit = benchmark_df['next_week_return']
        
        factor_dict= {'Jensen Alpha':[10],
                    'Annualized Returns':[10],
                    'Sharpe Ratio':[10],
                    'Information Ratio':[10]}
    
        end_week = set_end_week
        group_size = 5
        
    
        k = 0
        for car_num,topic_num in car_topic_pairs:
            for factor, filter_list in factor_dict.items():
                k+=1
                print(k)
                group_performance_function(car_num,topic_num,group_size,factor,filter_list,root_dir_1,root_dir_2,output_root_dir,benchmark_profit,end_week)
                print('***DONE***')



if __name__ == '__main__':
    Main()