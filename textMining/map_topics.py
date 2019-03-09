import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
import os
import json
import string
import nltk
from nltk.corpus import stopwords
from general_functions import collectFilenames


def loadJson(filepath_json,filename):
    '''
    Load contents of a json file.

    Args:
    filepath(str): filepath where the json file is in
    filename(str): filename of json file like 'AAPL_2014-01-01'

    Returns:
    contents(str): contents of json file
    '''
    with open(filepath_json+filename+'.json','r',encoding='utf-8') as json_file:
        contents=json.load(json_file)
    return contents

def saveContentsAsJson(contents,filepath,filename):
    with open(filepath+filename+'.json','w',encoding='utf-8') as json_file:
        json.dump(contents,json_file,ensure_ascii=False)


def saveJsonAsPickle(filepath_json,
                     pickle_json_full_name,
                     delete_repeated=True):
    '''
    This function is to collect all news of json files in "filepath_json" and save all news in the pickle file "pickle_json_full_name".

    Args:
    filepath_json(str): The filepath with json files we want to collect news from.
    pickle_json_full_name(str): The name of pickle file where news is saved in.
    delete_repeated(bool): If this is set as "True", then news with repeated "doc_id" would be deleted.

    Returns: All news of json files in "filepath_json"  would be saved in the pickle file "pickle_json_full_name".
    '''

    filename_list = collectFilename(filepath_json,'json')

    f = open(pickle_json_full_name,'wb')
    
    if not delete_repeated:
        for filename in filename_list:
            contents = loadJson(filepath_json,filename)
            last_update = contents['last_update']
            timezone = contents['timezone']
            news = contents['data']
            
            symbol = filename
            contents_dict_new = {"symbol":symbol,"last_update":last_update,"timezone":timezone,"data":news}
            pickle.dump(contents_dict_new,f)
    
    else:
        id_list = []
        
        for filename in filename_list:
            contents = loadJson(filepath_json,filename)
            
            last_update = contents['last_update']
            timezone = contents['timezone']

            data = contents['data']
            
            symbol = filename

            news_list = []

            for i in range(len(data)):
                news = data[i]
                id = news['doc_id']

                if id not in id_list:
                    id_list.append(id)
                    news_list.append(news)
            
            contents_dict_new = {"symbol":symbol,"last_update":last_update,"timezone":timezone,"data":news_list}
            pickle.dump(contents_dict_new,f)
    f.close()

    print('All json files have been saved in pickle file.')



def preprocessText(text,
                   stop_words_txt,
                   symbol_list):

    '''
    Precess text before doing IDA analysis.

    Args: 
    text(str): a single news

    Returns:
    filtered(list): a list of a news
    '''
    # Convert words into lowercase. For example, "Home" is converted to "home".  
    text = text.lower()
                                   
    # Remove '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~' from text
    for c in string.punctuation:                   
        text = text.replace(c, ' ')

    # Remove numbers 
    number_list = ['0','1','2','3','4','5','6','7','8','9']
    for i in number_list:
        text = text.replace(i," ")

    # Split sentence in a list of words
    wordLst = nltk.word_tokenize(text)              

    # Remove stopwords from text. 
    # Stopwords are words like "me","I","hasn't" and so on.
    wordLst = [w for w in wordLst if w not in stopwords.words('english')]    

    # Remove words with letters less than 3
    wordLst = [w for w in wordLst if len(w) >2]
    wordLst = [nltk.stem.WordNetLemmatizer().lemmatize(w, 'n') for w in wordLst]
    wordLst = [nltk.stem.WordNetLemmatizer().lemmatize(w, 'v') for w in wordLst]
    wordLst = [w for w in wordLst if len(w) > 2]
    wordLst = [w for w in wordLst if w not in stopwords.words('english')] 

    self_stop_words = []
    with open(stop_words_txt,'r') as f:
        for line in f:
            if line != '':
                self_stop_words.append(line.strip().lower())
            
    wordLst = [w for w in wordLst if w not in self_stop_words]
    
    if symbol_list != None:
        wordLst = [w if w not in symbol_list else 'symbolmask' for w in wordLst]


    return ' '.join(wordLst)



def collectFilename(filepath_files,filetype='csv'):
    '''
    Collect names of files in specific dirs.
    
    Args:
        filepath(str): Filepath we want to collect stock tickers.
        
    Returns:
        file_list(list): List of tickers that have CSV in specific filepath.
    '''
    filename_list=[]
    for root, dirs, files in os.walk(filepath_files):
        if files:
            for f in files:
                if filetype in f:
                    filename_list.append(f.split('.'+filetype)[0])
    return filename_list


def addPreprocessNewsToPickle(pickle_json_full_name,
                              stop_words_txt,
                              symbol_list,
                              pickle_with_process_news):
    '''
    For each single piece of news, add preprocessed to the original dict of news and save new dicts to a new pickle file.
    
    Args:
        pickle_json_full_name(str): full name of pickle file which entails the contents of all raw news.
        stop_word_txt(str): full name of txt file which entails the additional stop words that need to be removed from news.
        symbol_list(str): full name of dataFrame file which entails all symbols that need to be substituted with "symbolmask" in news.
        pickle_with_processs_news(str): full name of pickle file in which the news with preprocess news are saved.
    
    '''
    
    f_json = open(pickle_json_full_name,'rb')
    
    f_output = open(pickle_with_process_news,'wb')


    x = 0
    while True:
        try:
            contents = pickle.load(f_json)
        
        except:
            break
        
        else:
            x+=1

            last_update = contents['last_update']
            timezone = contents['timezone']
            news = contents['data']
            symbol = contents['symbol']
            
            print(x)  
            print(str(symbol)+'   ----')          

            for i in range(len(news)):
                text = news[i]['teaser']
                preprocess_news_str = preprocessText(text,stop_words_txt,symbol_list)
                news[i]['precess_news'] = preprocess_news_str
            
            contents_dict_new = {"symbol":symbol,"last_update":last_update,"timezone":timezone,"data":news}

            pickle.dump(contents_dict_new,f_output)
        
    f_json.close()
    f_output.close()

    print('All news in '+str(pickle_json_full_name)+' has been processed to '+str(pickle_with_process_news))



def mapTopics(lda,
              docLst_all,
              fix_max_df,
              fix_min_df,
              fix_max_features,
              fix_stop_words):
    '''
    For each single news, map the news with lda model and get the mapped topics.
    
    Args:
        lda(model): lda model from training the news related to high abnormal returns.
        docLst_all(list): a list which entails all preprocessed news
        fix_max_df(int),fix_min_df(int),fix_max_features(int),fix_stop_words(str): parameters of  lda model
    
    Returns:
        topics_all_data_list(list): a list which entails data related to mapping topics for all preprocessed news.
                                    This is a list of multi lists.
        topics_num_list(list): a list which entails topics with largest correlation in mapping news to topics.
                               This is a list of multi numbers.
    '''
    
    tf_vectorizer = CountVectorizer(max_df=fix_max_df, 
                                    min_df=fix_min_df,
                                    max_features=fix_max_features,
                                    stop_words=fix_stop_words)
    
    X_test = tf_vectorizer.fit_transform(docLst_all)
    map_results = lda.transform(X_test)

    topics_num_list = []
    topics_all_data_list = []

    for data in map_results:
        topics_all_data_list.append(list(data))
        topic_possible = data.max()
        topics_num_list.append(str(list(data).index(topic_possible)+1))
    print('Topics mapping finished')
    return topics_all_data_list,topics_num_list



def addTopicsToPickle(pickle_with_process_news,pickle_news_with_topics,topics_all_data_list,topics_num_list):
    
    '''
    For each single piece of news, add topics to the original dict of news and save new dicts to a new pickle file.
    
    Args:
        pickle_with_processs_news(str): full name of pickle file in which the news with preprocess news are saved.
                                        This pickle file entails input details.
        pickle_news_with_topics(str):full name of pickle file in which the news with both preprocess news and topics are saved.
                                     This pickle file entails output details.
        topics_all_data_list(list): a list which entails data related to mapping topics for all preprocessed news.
                                    This is a list of multi lists.
        topics_num_list(list): a list which entails topics with largest correlation in mapping news to topics.
                               This is a list of multi numbers.
        summary_csv_output(str): full name of csv file in which details of news and topics are saved.
                                 This csv file entails output details.

    '''
    
    f_input = open(pickle_with_process_news,'rb')
    f_output = open(pickle_news_with_topics,'wb')

    x = 0
    original_total_number = len(topics_num_list)
    print(original_total_number)

    print('Start to map all topics ')
        
    while True:
        try:
            contents = pickle.load(f_input)
        
        except:
            break
        
        else:      
            last_update = contents['last_update']
            timezone = contents['timezone']
            news = contents['data']
            symbol = contents['symbol']
           
            
            for i in range(len(news)):
                x+=1
                
                news[i]['topic_number'] = topics_num_list.pop(0)
                news[i]['topic_data'] = topics_all_data_list.pop(0)
                
            contents_dict_new = {"symbol":symbol,"last_update":last_update,"timezone":timezone,"data":news}
            pickle.dump(contents_dict_new,f_output)
                
    print('Mapping done')
    print('Total Number of news with topics -> '+str(original_total_number))
    print('Total number of news '+str(x))
    
    f_input.close()
    f_output.close()



def getDoclstFromPickle(pickle_with_process_news):  
    '''
    Combine preprocess news in pickle file to a list.
    
    Args:
        pickle_with_processs_news(str): full name of pickle file in which the news with preprocess news are saved.
                                        This pickle file entails input details.
    
    Returns:
        docLst_all(list): a list which entails all preprocessed news.
                          This is a list of multi strings.
                          For each string in the list, there are multi words after preprocess. 
    '''
    docLst_all = []

    f = open(pickle_with_process_news,'rb')

    while True:
        try:
            contents = pickle.load(f)

        except:
            break

        else:
            news = contents['data']

            for i in range(len(news)):

                docLst_all.append(news[i]['precess_news'])

    f.close()
    print('Combine finished---')

    return docLst_all


def getTopicsSummary(topic_num,pickle_news_with_topics, summary_csv):
    f = open(pickle_news_with_topics,'rb')
    
    pd.DataFrame().to_csv(summary_csv)
    print('Start to collect details from pickle to csv')
    
    x=0
    topic_col = ['topic'+str(k+1) for k in range(topic_num)]
    symbol_list = []
    date_list = []
    sentiment_list=[]
    relevance_list = []
    topic_num_list = []
    topic_col = ['topic'+str(k+1) for k in range(topic_num)]
    topic_data_dict = {t:[] for t in topic_col}
    
    while True:
        if x%10000 <10:
            print('News No.'+str(x))
            
        try:
            contents = pickle.load(f)
        except:
            break
        else:
            news = contents['data']
            symbol = contents['symbol']
            
            for single_news in news:
                x+=1
                symbol_list.append(symbol)
                date_list.append(single_news['date'])
                sentiment_list.append(single_news['sentiment'])
                relevance_list.append(single_news['relevance'])
                topic_num_list.append(single_news['topic_number'])
                topic_data = single_news['topic_data']
                for j in range(topic_num):
                    topic_c = topic_col[j]
                    topic_data_dict[topic_c].append(topic_data[j])
    
    temp_df = pd.DataFrame()
    temp_df['symbol'] = pd.Series(symbol_list)
    temp_df['Date'] = pd.Series(date_list)
    temp_df['sentiment'] = pd.Series(sentiment_list)
    temp_df['relevance'] =pd.Series( relevance_list)
    temp_df['topic_number'] = pd.Series(topic_num_list)
    for j in range(topic_num):
        topic_c = topic_col[j]
        temp_df[topic_c] = pd.Series(topic_data_dict[topic_c])
    output = pd.read_csv(summary_csv,index_col=0)
    output = pd.concat([output,temp_df],ignore_index = True)
    output.to_csv(summary_csv)
    print('csv saved '+str(x))
    f.close()
    print('ALL DONE---------------------------------')




def mapping_main_function(lda_class_pickle,pickle_with_process_news,n_topic,car_num,load_lda_j,fix_features,output_dir,single_year):

    filepath_temp =str(car_num)+'_car_'+str(n_topic)+'_topics/'
    pickle_news_with_topics = output_dir+str(filepath_temp)+str(single_year)+'_all_news_'+str(car_num)+'car_with_'+str(n_topic)+'topics.pickle'
    summary_csv = output_dir+str(filepath_temp)+str(single_year)+'summary_all_news_'+str(car_num)+'car_'+str(n_topic)+'topics_full.csv'
    
    docLst_all = getDoclstFromPickle(pickle_with_process_news)

    f_lda = open(lda_class_pickle,'rb')
    for i in range(load_lda_j):
        lda_model = pickle.load(f_lda)
    f_lda.close()

    topics_all_data_list,topics_num_list = mapTopics(lda_model,docLst_all,fix_max_df = 0.95,fix_min_df = 2,fix_max_features = fix_features,fix_stop_words = 'english')
    addTopicsToPickle(pickle_with_process_news,pickle_news_with_topics,topics_all_data_list,topics_num_list)
    getTopicsSummary(n_topic,pickle_news_with_topics, summary_csv)
    print('Done >>>>>>> '+str(n_topic))



def saveRawNewsAsPickle(dir_name, pickle_name):
    files = collectFilenames(dir_name, 'json')
    pickle_f = open(pickle_name,'wb')
    for f in files:
        symbol = f
        contents = loadJson(dir_name,f)
        contents['symbol'] = symbol
        
        pickle.dump(contents, pickle_f)
    pickle_f.close()

'''



def _mapping():
    pickle_with_process_news = '/home/zhangwei/output-topics/input/2017_all_news_with_process_news.pickle'

    output_dir = '/home/zhangwei/2018-07-25/2017_in_sample/'
    single_year = 2017

    lda_class_pickle = '/home/zhangwei/2018-07-25/2018_out_of_sample/lda_20car_20180725.pickle'
    # 20_car_55_topics
    n_topic = 50
    car_num = 20
    load_lda_j = 7
    fix_features = 6993
    mapping_main_function(lda_class_pickle,pickle_with_process_news,n_topic,car_num,load_lda_j,fix_features,output_dir,single_year)





#20_Car_50_topics

def fifty_topics_in_sample():
    pickle_with_process_news = '/home/zhangwei/output-topics/input/2017_all_news_with_process_news.pickle'

    output_dir = '/home/zhangwei/2018-07-25/2017_in_sample/'
    single_year = 2017

    lda_class_pickle = '/home/zhangwei/2018-07-25/2018_out_of_sample/lda_20car_20180725.pickle'
    # 20_car_55_topics
    n_topic = 50
    car_num = 20
    load_lda_j = 7
    fix_features = 6993
    main_function(lda_class_pickle,pickle_with_process_news,n_topic,car_num,load_lda_j,fix_features,output_dir,single_year)


def fifty_topics_out_of_sample():
    pickle_with_process_news = '/home/zhangwei/output-topics/input/2018_all_news_with_process_news.pickle'

    output_dir = '/home/zhangwei/2018-07-25/2018_out_of_sample/'
    single_year = 2018

    lda_class_pickle = '/home/zhangwei/2018-07-25/2018_out_of_sample/lda_20car_20180725.pickle'
    # 20_car_55_topics
    n_topic = 50
    car_num = 20
    load_lda_j = 7
    fix_features = 6993
    main_function(lda_class_pickle,pickle_with_process_news,n_topic,car_num,load_lda_j,fix_features,output_dir,single_year)



if __name__ == '__main__':
    fifty_topics_in_sample()
    fifty_topics_out_of_sample()


'''














    


def in_sample():
    
    stop_words_txt = 'D:/data/input/stop_words.txt'
    symbol_series = pd.read_csv('D:/data/input/downloadData.py_input/companylist.csv')['Symbol']
    symbol_list = [symbol.lower() for symbol in symbol_series ]

    # 2015: in-sample, 2016: out-of-sample
    prior = 2015
    post = 2016
    base_dir = 'D:/data/strategy_data/{}-{}/'.format(prior, post)

    single_year = prior
    
    news_dir = 'D:/data/news_{}/'.format(single_year)
    pickle_json_full_name = base_dir + 'news_{}_all.pickle'.format(single_year)

    #saveRawNewsAsPickle(news_dir, pickle_json_full_name)
    pickle_with_process_news = base_dir + 'news_{}_process.pickle'.format(single_year)
    output_dir = base_dir +'{}_in_sample/'.format(single_year)
    
    #addPreprocessNewsToPickle(pickle_json_full_name,stop_words_txt,symbol_list,pickle_with_process_news)


    
    #########################
    # 10_car_20_topics
    n_topic = 20
    car_num = 10
    load_lda_j = int((n_topic-20)/5 + 1)
    fix_features = 15535
    lda_class_pickle = base_dir + '{}_car_{}_preprocess_lda.pickle'.format(car_num,prior)
    mapping_main_function(lda_class_pickle,pickle_with_process_news,n_topic,car_num,load_lda_j,fix_features,output_dir,single_year)

    
    #########################
    # 10_car_30_topics
    n_topic = 30
    car_num = 10
    load_lda_j = int((n_topic-20)/5 + 1)
    fix_features = 15535
    lda_class_pickle = base_dir + '{}_car_{}_preprocess_lda.pickle'.format(car_num,prior)

    mapping_main_function(lda_class_pickle,pickle_with_process_news,n_topic,car_num,load_lda_j,fix_features,output_dir,single_year)

    
    #########################
    # 20_car_20_topics
    n_topic = 20
    car_num = 20
    load_lda_j = int((n_topic-20)/5 + 1)
    fix_features = 11828
    lda_class_pickle = base_dir + '{}_car_{}_preprocess_lda.pickle'.format(car_num,prior)

    mapping_main_function(lda_class_pickle,pickle_with_process_news,n_topic,car_num,load_lda_j,fix_features,output_dir,single_year)

    
    #########################
   # 20_car_20_topics
    n_topic = 40
    car_num = 20
    load_lda_j = int((n_topic-20)/5 + 1)
    fix_features = 11828
    lda_class_pickle = base_dir + '{}_car_{}_preprocess_lda.pickle'.format(car_num,prior)

    mapping_main_function(lda_class_pickle,pickle_with_process_news,n_topic,car_num,load_lda_j,fix_features,output_dir,single_year)
    



def out_of_sample():
    stop_words_txt = 'D:/data/input/stop_words.txt'
    symbol_series = pd.read_csv('D:/data/input/downloadData.py_input/companylist.csv')['Symbol']
    symbol_list = [symbol.lower() for symbol in symbol_series ]

    # 2015: in-sample, 2016: out-of-sample
    prior = 2015
    post = 2016
    base_dir = 'D:/data/strategy_data/{}-{}/'.format(prior, post)

    single_year = post
    news_dir = 'D:/data/news_{}/'.format(single_year)
    pickle_json_full_name = base_dir + 'news_{}_all.pickle'.format(single_year)

    #saveRawNewsAsPickle(news_dir, pickle_json_full_name)
 
    pickle_with_process_news = base_dir + 'news_{}_process.pickle'.format(single_year)
    output_dir = base_dir +'{}_out_of_sample/'.format(single_year)
    
    #addPreprocessNewsToPickle(pickle_json_full_name,stop_words_txt,symbol_list,pickle_with_process_news)

    
    
    #########################
    # 10_car_20_topics
    n_topic = 20
    car_num = 10
    load_lda_j = int((n_topic-20)/5 + 1)
    fix_features = 15535
    
    lda_class_pickle = base_dir + '{}_car_{}_preprocess_lda.pickle'.format(car_num,prior)

    mapping_main_function(lda_class_pickle,pickle_with_process_news,n_topic,car_num,load_lda_j,fix_features,output_dir,single_year)
    
    #########################
    # 10_car_30_topics
    n_topic = 30
    car_num = 10
    load_lda_j = int((n_topic-20)/5 + 1)
    fix_features = 15535
    lda_class_pickle = base_dir + '{}_car_{}_preprocess_lda.pickle'.format(car_num,prior)
    mapping_main_function(lda_class_pickle,pickle_with_process_news,n_topic,car_num,load_lda_j,fix_features,output_dir,single_year)
    
    
    #########################
    # 20_car_20_topics
    n_topic = 20
    car_num = 20
    load_lda_j = int((n_topic-20)/5 + 1)
    fix_features = 11828

    lda_class_pickle = base_dir + '{}_car_{}_preprocess_lda.pickle'.format(car_num,prior)

    mapping_main_function(lda_class_pickle,pickle_with_process_news,n_topic,car_num,load_lda_j,fix_features,output_dir,single_year)

    
    #########################
   # 20_car_20_topics
    n_topic = 40
    car_num = 20
    load_lda_j = int((n_topic-20)/5 + 1)
    fix_features = 11828


    lda_class_pickle = base_dir + '{}_car_{}_preprocess_lda.pickle'.format(car_num,prior)

    mapping_main_function(lda_class_pickle,pickle_with_process_news,n_topic,car_num,load_lda_j,fix_features,output_dir,single_year)
    
    
    
    
    
def in_sample_2():
    
    stop_words_txt = 'D:/data/input/stop_words.txt'
    symbol_series = pd.read_csv('D:/data/input/downloadData.py_input/companylist.csv')['Symbol']
    symbol_list = [symbol.lower() for symbol in symbol_series ]

    # 2016: in-sample, 2017: out-of-sample
    prior = 2016
    post = 2017
    base_dir = 'D:/data/strategy_data/{}-{}/'.format(prior, post)

    single_year = prior
    
    news_dir = 'D:/data/news_{}/'.format(single_year)
    pickle_json_full_name = base_dir + 'news_{}_all.pickle'.format(single_year)

    #saveRawNewsAsPickle(news_dir, pickle_json_full_name)
    pickle_with_process_news = base_dir + 'news_{}_process.pickle'.format(single_year)
    output_dir = base_dir +'{}_in_sample/'.format(single_year)
    
    #addPreprocessNewsToPickle(pickle_json_full_name,stop_words_txt,symbol_list,pickle_with_process_news)


    '''
    #########################
    # 10_car_20_topics
    n_topic = 20
    car_num = 10
    load_lda_j = int((n_topic-20)/5 + 1)
    fix_features = 14710
    lda_class_pickle = base_dir + '{}_car_{}_preprocess_lda.pickle'.format(car_num,prior)
    mapping_main_function(lda_class_pickle,pickle_with_process_news,n_topic,car_num,load_lda_j,fix_features,output_dir,single_year)

    '''
    #########################
    # 10_car_40_topics
    n_topic = 40
    car_num = 10
    load_lda_j = int((n_topic-25)/5 + 1)
    fix_features = 14710
    lda_class_pickle = base_dir + '{}_car_{}_preprocess_lda.pickle'.format(car_num,prior)

    mapping_main_function(lda_class_pickle,pickle_with_process_news,n_topic,car_num,load_lda_j,fix_features,output_dir,single_year)

    '''
    #########################
    # 20_car_20_topics
    n_topic = 20
    car_num = 20
    load_lda_j = int((n_topic-20)/5 + 1)
    fix_features = 10877
    lda_class_pickle = base_dir + '{}_car_{}_preprocess_lda.pickle'.format(car_num,prior)

    mapping_main_function(lda_class_pickle,pickle_with_process_news,n_topic,car_num,load_lda_j,fix_features,output_dir,single_year)

    '''
    #########################
   # 20_car_20_topics
    n_topic = 40
    car_num = 20
    load_lda_j = int((n_topic-25)/5 + 1)
    fix_features = 10877
    lda_class_pickle = base_dir + '{}_car_{}_preprocess_lda.pickle'.format(car_num,prior)

    mapping_main_function(lda_class_pickle,pickle_with_process_news,n_topic,car_num,load_lda_j,fix_features,output_dir,single_year)




def out_of_sample_2():
    stop_words_txt = 'D:/data/input/stop_words.txt'
    symbol_series = pd.read_csv('D:/data/input/downloadData.py_input/companylist.csv')['Symbol']
    symbol_list = [symbol.lower() for symbol in symbol_series ]

    # 2016: in-sample, 2017: out-of-sample
    prior = 2016
    post = 2017
    base_dir = 'D:/data/strategy_data/{}-{}/'.format(prior, post)

    single_year = post
    news_dir = 'D:/data/news_{}/'.format(single_year)
    pickle_json_full_name = base_dir + 'news_{}_all.pickle'.format(single_year)

    #saveRawNewsAsPickle(news_dir, pickle_json_full_name)
 
    pickle_with_process_news = base_dir + 'news_{}_process.pickle'.format(single_year)
    output_dir = base_dir +'{}_out_of_sample/'.format(single_year)
    
    #addPreprocessNewsToPickle(pickle_json_full_name,stop_words_txt,symbol_list,pickle_with_process_news)

    
    '''
    #########################
    # 10_car_20_topics
    n_topic = 20
    car_num = 10
    load_lda_j = int((n_topic-20)/5 + 1)
    fix_features = 14710
    
    lda_class_pickle = base_dir + '{}_car_{}_preprocess_lda.pickle'.format(car_num,prior)

    mapping_main_function(lda_class_pickle,pickle_with_process_news,n_topic,car_num,load_lda_j,fix_features,output_dir,single_year)

    #########################
    # 10_car_40_topics
    n_topic = 40
    car_num = 10
    load_lda_j = int((n_topic-25)/5 + 1)
    fix_features = 14710
    lda_class_pickle = base_dir + '{}_car_{}_preprocess_lda.pickle'.format(car_num,prior)
    mapping_main_function(lda_class_pickle,pickle_with_process_news,n_topic,car_num,load_lda_j,fix_features,output_dir,single_year)
    
    #########################
    # 20_car_20_topics
    n_topic = 20
    car_num = 20
    load_lda_j = int((n_topic-20)/5 + 1)
    fix_features = 10877

    lda_class_pickle = base_dir + '{}_car_{}_preprocess_lda.pickle'.format(car_num,prior)

    mapping_main_function(lda_class_pickle,pickle_with_process_news,n_topic,car_num,load_lda_j,fix_features,output_dir,single_year)

    '''
    #########################
   # 20_car_20_topics
    n_topic = 40
    car_num = 20
    load_lda_j = int((n_topic-25)/5 + 1)
    fix_features = 10877


    lda_class_pickle = base_dir + '{}_car_{}_preprocess_lda.pickle'.format(car_num,prior)

    mapping_main_function(lda_class_pickle,pickle_with_process_news,n_topic,car_num,load_lda_j,fix_features,output_dir,single_year)

#================================================================================

if __name__ == '__main__':

    out_of_sample_2()