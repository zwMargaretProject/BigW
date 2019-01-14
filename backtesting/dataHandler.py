import pandas as pd
import json

from constant import LOCAL_CSV, LOCAL_MONGODB, CLOUD_YAHOO, CLOUD_IEX, CLOUD_GOOGLE
from generalFunction import getSetting

class DataHandler(object):
    
    def __init__(self, csvDir, symbolList):
        self.localDataSource = LOCAL_CSV
        self.csvDir = csvDir
        self.symbolList = symbolList
        self.symbolData = {}
        self.latestSymbolData = {}

    def _openConvertCsvFiles(self):
        mergeIndex = None

        for symbol in self.symbolList:
            df = pd.read_csv(os.path.join(self.csvDir, '{}.csv'.format(symbol)))

            # Change column names
            #### ADD COMMANT HERE

            self.symbolData[symbol] = df
            if mergeIndex is None:
                mergeIndex = df.index
            else:
                mergeIndex.union(df.index)
            self.latestSymbolData[symbol] = []
        
        for symbol in self.symbolList:
            self.symbolData[symbol] = self.symbolData[symbol].reindex(index=mergeIndex, method='pad').iterrows()
        self.localEndDate = mergeIndex[-1]
    
    def _updateLocalData(self, cloudDataSource, startDate, endDate, symbolList=None):
        if symbolList is None:
            symbolList = self.symbolList
        cloudFunc = {'google':googleFunc, 'yahoo':yahooFunc, 'iex':iexFunc, 'quandl':quandlFunc}
        downloadFunc = cloudFunc[cloudDataSource]
        print("Start to update data from {}. Number of symbols: {}; End Date: {}".format(cloudDataSource, len(symbolList), endDate))
        cloudFunc(startDate, endDate, symbolList, self.csvDir)
        print("Update Finished. Data is saved in {}.".format(self.csvDir))

    
    def newBar(self):
        pass
        

