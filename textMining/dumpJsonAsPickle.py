import pandas as pd
import numpy as np
from urllib import request
import json
import time
import pickle
import os

    
#################################################
# Here are functions for loading and combining json and txt files.

#-----------------------------------------------

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


#-------------------------------------------------

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

#------------------------------------------------

def saveContentsAsJson(contents,filepath,filename):
    with open(filepath+filename+'.json','w',encoding='utf-8') as json_file:
        json.dump(contents,json_file,ensure_ascii=False)
        
#---------------------------------------------------

def saveJsonAsPickle(filepath_json,pickle_json_full_name,delete_repeated=True):
    '''
    This function is to collect all news of json files in "filepath_json" and save all news in the pickle file "pickle_json_full_name".

    Args:
    filepath_json(str): The filepath with json files we want to collect news from.
    pickle_json_full_name(str): The name of pickle file where news is saved in.
    delete_repeated(bool): If this is set as "True", then news with repeated "doc_id" would be deleted.

    Returns: All news of json files in "filepath_json"  would be saved in the pickle file "pickle_json_full_name".
    '''

    filename_list = collectFilename(filepath_json,'json')
    dict_list = []

    f = open(pickle_json_full_name,'wb')
    
    if not delete_repeated:
        for filename in filename_list:
            contents = loadJson(filepath_json,filename)
            last_update = contents['last_update']
            time_zone = contents['timezone']
            news = contents['data']
            
            symbol = filename[:-11]
            event_date = filename[-10:]
            contents_dict_new = {"symbol":symbol,"event_date":event_date,"last_update":last_update,"time_zone":time_zone,"data":news}
            pickle.dump(contents_dict_new,f)
    
    else:
        id_list = []
        
        for filename in filename_list:
            contents = loadJson(filepath_json,filename)
            
            last_update = contents['last_update']
            time_zone = contents['timezone']

            data = contents['data']
            
            symbol = filename[:-11]
            event_date = filename[-10:]

            news_list = []

            for i in range(len(data)):
                news = data[i]
                id = news['doc_id']

                if id not in id_list:
                    id_list.append(id)
                    news_list.append(news)
            
            contents_dict_new = {"symbol":symbol,"event_date":event_date,"last_update":last_update,"time_zone":time_zone,"data":news_list}
            pickle.dump(contents_dict_new,f)
    f.close()

    print('All json files have been saved in pickle file.')



##################################################


############################################################################   


def main():

    filepath_json_1 = '/data/output_news/news_10car/'
    filepath_json_2 = '/data/output_news/news_20car/'

    pickle_json_full_name_1 = '/data/output_news/raw_news_10car.pickle'
    pickle_json_full_name_2 = '/data/output_news/raw_news_20car.pickle'

    saveJsonAsPickle(filepath_json_1,pickle_json_full_name_1,delete_repeated=True)
    saveJsonAsPickle(filepath_json_2,pickle_json_full_name_2,delete_repeated=True)

if __name__ == '__main__':
    main()

