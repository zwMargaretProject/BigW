import pandas as pd
import os
import numpy as np
import json

###########################################################
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



###########################################################

def check_and_create_dir(path):
    isExists=os.path.exists(path)
    if not isExists:
        os.makedirs(path) 
        print ('{} is built successfully.'.format(path))
    else:
        print ('{} exists.'.format(path))


#############
def collectFilenames(filepath, type = 'csv'):
    '''
    Collect stock symbols with price data in specific dirs.
    
    Args:
        filepath(str): Filepath we want to collect stock tickers.
        
    Returns:
        stock_list(list): List of tickers that have CSV in specific filepath.
    '''
    files_list=[]
    for root, dirs, files in os.walk(filepath):
        if files:
            for f in files:
                if type in f:
                    files_list.append(f.split('.'+type)[0])
    return files_list

    
def saveContentsAsJson(contents,filepath,filename):
    with open(filepath+filename+'.json','w',encoding='utf-8') as json_file:
        json.dump(contents,json_file,ensure_ascii=False)
        
def loadJson(filepath,filename):
    with open(filepath+filename+'.json','r',encoding='utf-8') as json_file:
        contents=json.load(json_file)
    return contents