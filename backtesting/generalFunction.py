import pandas as pd
import os
import numpy as np
import json

#------------------------------------------------
def collectFileName(filepath, filetype='csv'):
    '''
    Collect filenames in specific dirs.
    Args:
        filepath(str): Filepath we want to collect stock tickers.   
    Returns:
        filename_list(list): List of tickers that have CSV in specific filepath.
    '''
    fileNames = []
    for root, dirs, files in os.walk(filepath):
        if files:
            for f in files:
                if filetype in f:
                    fileNames.append(f.split(filetype)[0])
    return fileNames

#------------------------------------------------
def checkAndCreateDir(path):
    isExists=os.path.exists(path)
    if not isExists:
        os.makedirs(path) 
        print ('{} is built successfully.'.format(path))
    else:
        print ('{} exists.'.format(path))