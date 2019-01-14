import numpy as np
import talib
from constant import *
from eventObject import BarEvent
from backtestingEngine import BacktestingEngine

class CtaStrategy(object):
    inited = False
    trading = False
    position = 0
    strategyName = 'cta_strategy_template'
    paramList = ['strategyName']

    def __init__(self, backtestEngineClass, setting={}):
        self.backtestEngineClass = backtestEngineClass
        if setting:
            d = self.__dict__
            for key in self.paramList:
                if key in setting:
                    d[key] = setting[key]

    def initialize(self):
    	'''
    	Please re-write this function for strategy needs.
    	'''
        raise NotImplementedError
    
    def fromTickToOrder(self):
    	'''
    	Please re-write this function for strategy needs.
    	'''
        raise NotImplementedError
    
    def fromBarToOrder(self):
    	'''
    	Please re-write this function for strategy needs.
    	'''
        raise NotImplementedError

    def x_fromBarToOrder(self):
    	'''
    	Please re-write this function for strategy needs.
    	'''
        raise NotImplementedError


    #----------------------------------------------------
    # Functions below can be directly made used of.

    def start(self):
        self.writeLog('Strategy is initialized.')
    
    def stop(self):
        self.writeLog('Strategy is stopped.')
    
    def buy(self, price, volume, stop=False):
        return self.sendOrder(CTAORDER_BUY, price, volume, stop)

    def sell(self, price, volume, stop=False):
        return self.sendOrder(CTAORDER_SELL, price, volume, stop)

    def short(self, price, volume, stop=False):
        return self.sendOrder(CTAORDER_SHORT, price, volume, stop)

    def cover(self, price, volume, stop=False):
        return self.sendOrder(CTAORDER_COVER, price, volume, stop)
    
    def sendOrder(self, orderType, price, volume, stop=False):
        if self.trading:
            if not stop:
                orderIDList = self.backtestEngineClass.sendOrder(self.symbol, orderType, price, volume, self)
            else:
                orderIDList = self.backtestEngineClass.sendStopOrder(self.symbol, orderType, price, volume, self)
            return orderIDList
        return []
    
    def cancelOrder(self, orderID):
        if not orderID:
            return
        if STOPORDERPREFIX in orderID:
            self.backtestEngineClass.cancelStopOrder(orderID)
        else:
            self.backtestEngineClass.cancelOrder(orderID)
    
    def cancelAllOrder(self):
        self.backtestEngineClass.cancelAllOrder(self.name)

    def getInitData(self):
    	return self.backtestEngineClass.getInitData()
    
    def writeLog(self, content):
    	content = self.name + ': ' + content
    	self.backtestEngineClass.writeLog(content)

    

    
         