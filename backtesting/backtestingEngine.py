from collections import OrderedDict
from eventType import *
from eventObject import (tickEvent, barEvent, orderEvent, tradeEvent)
from constants import (EMPTY_STRING, EMPTY_INT)
from datetime import (datetime, timedelta)
from dataHandler import DataHandler


##---------------------------------------------------
class BacktestingEngine(object):
    BAR_MODE = 'bar'
    TICK_MODE = 'tick'
 
    #################################################
    def __init__(self):
        
        self.strategyClass = None
        self.symbol = ''
        self.csvDir = ''
        self.dataClass = None

        self.mode = self.BAR_MODE

        self.marketOrderCount = 0
        self.marketOrderDict = OrderedDict()
        self.workingMarketOrderDict = OrderedDict()

        self.limitOrderCount = 0
        self.limitOrderDict = OrderedDict()
        self.workingLimitOrderDict = OrderedDict()

        self.stopOrderCount = 0
        self.limitOrderDict = OrderedDict()
        self.workingLimitOrderDict = OrderedDict()

        self.startDate = ''
        self.initDays = 0
        self.endDate = ''

        self.capital = 1000000
        self.slippage = 0
        self.rate = 0
        self.size = 1
        self.priceTick = 0

        self.initData = []

        self.dataStartDate = None
        self.strategyStartDate = None
        self.dataEndDate = None

        self.tradeCount = 0
        self.tradeDict = OrderedDict()

        self.logList = []
        
        self.latestBar = None
        self.latestTick = None
        self.latestDatetime = None

        self.dailyResultDict = OrderedDict()

    #################################################
    def output(self, content):
        print(str(datetime.now()) + '\t' + content)

    def writeLog(self, content):
        pass

    #################################################
    def setMultiDates(self, startDate='20101231', initDays=10, endDate=''):
        self.startDate = dataStartDate
        self.initDays = initDays
        self.endDate = endDate
        
        self.dataStartDate = datetime.strptime(startDate, '%Y%m%d')
        self.strategyStartDate = self.dataStartDate + timedelta(initDays)

        if endDate:
            self.dataEndDate = datetime.strptime(endDate, '%Y%m%d')
            self.dataEndDate = self.dataEndDate.replace(hour=23, minute=59)

    def setStrategyClass(self, strategyClass):
        self.strategyClass = strategyClass
    
    def setDataClass(self, csvDir, symbol):
        self.csvDir = csvDir
        self.symbol = symbol
        self.dataClass = dataHandler([symbol], csvDir)
    
    def setMode(self, mode):
        self.mode = mode

    def setCapital(self, capital):
        self.capital = capital
    
    def setSlippage(self, slippage):
        self.slippage = slippage
    
    def setSize(self, size):
        self.size = size
    
    def setRate(self, rate):
        self.rate = rate

    #################################################
    def loadHistoryData(self):
        self.initData = []
        if self.mode == self.BAR_MODE:
            iterrow_list = self.dataClass.loadCsvBar(self.dataStartDate, self.strategyStartDate, includeEnd=False)
            for barData in iterrow_list:
                singleBar = BarEvent()
                singleBar.__dict__ = barData
                self.initData.append(singleBar)
            self.strategyData = self.dataClass.loadCsvBar(self.strategyStartDate, self.dataEndDate, includeEnd=True)
    
        if self.mode == self.TICK_MODE:
            iterrow_list = self.dataClass.loadCsvTick(self.dataStartDate, self.strategyStartDate, includeEnd=False)
            for tickData in iterrow_list:
                singleTick = TickEvent()
                singleTick.__dict__ = tickData
                self.initData.append(singleTick)
            self.strategyData = self.dataClass.loadCsvTick(self.strategyStartDate, self.dataEndDate, includeEnd=True)
    
        self.output('Finish: loading histroy data')

    #################################################
    def runBacktesting(self):
        self.loadHistroyData()
        self.output('Backtesting starts.')

        self.strategyClass.inited = True
        self.strategyClass.initalize()
        self.output('Strategy has been initalized.')

        self.strategyClass.trading = True
        self.strategyClass.start()
        self.output('Strategy starts.')

        self.output('Start to load strategy data.')
        
        for row in self.strategyData:
            if self.mode == self.BAR_MODE:
                singleBar = BarEvent()
                singleBar.__dict__ = dict(row)
                self.processNewMode(singleBar)

            elif self.mode == self.TICK_MODE:
                singleTick = TickEvent()
                singleTick.__dict__ = dict(row)
                self.processNewTick(singleTick)

        self.output("Finish to load strategy data.")

    #################################################
    def processNewBar(self, bar_event):
        self.latestBar = bar_event
        self.latestDatetime = bar_event.datetime

        self.processLimitOrder()
        self.processStopOrder()
        self.strategyClass.onBar(bar_event)
        self.updateDailyClose(bar_event.datetime, bar_event.close)

    def processNewTick(self, tick_event):
        self.latestTick = tick_event
        self.latestDatetime = tick_event.datetime

        self.processLimitOrder()
        self.processStopOrder()
        self.strategyClass.onTick(tick_event)
        self.updateDailyClose(tick_event.datetime, tick_event.close)

    #################################################
    def processLimitOrder(self):
        if self.mode == self.BAR_MODE:
            buyCrossPrice = self.latestBar.low
            sellCrossPrice = self.latestBar.high
            buyBestPrice = self.latestBar.open
            sellBestPrice = self.latestBar.open

        elif self.mode == self.TICK_MODE:
            buyCrossPrice = self.latestTick.askPrice1
            sellCrossPrice = self.latestTick.bidPrice1
            buyBestPrice = self.latestTick.askPrice1
            sellBestPrice = self.latestTick.bidPrice1

        for orderID, order_event in self.workingLimitOrderDict.items():
            if order_event.status is None:
                order_event.status = STATUS_NOTTRADED

            buySuccessed = (order_event.direction == DIRECTION_LONG and order_event.price>=buyCrossPrice and buyCrossPrice>=0)
            sellSuccessed = (order_event.direction == DIRECTION_SHORT and order_event.price<=sellCrossPrice and sellCrossPrice<=0)

            if buySuccessed or sellSuccessed:
                self.tradeCount += 1
                tradeID = str(self.tradeCount)
                trade_event = TradeEvent()
                trade_event.tradeID = tradeID
                trade_event.orderID = orderID
                trade_event.direction = order_event.direction
                trade_event.offset = order_event.offset

                if buySuccessed:
                    trade_event.price = min(order_event.price, buyBestCrossPrice)
                    self.strategyClass.position += order_event.totalVolume
                elif sellSuccessed:
                    trade_event.price = max(order_event.price, sellBestCrossPrice)
                    self.strategyClass.postion -= order_event.totalVolume
                trade_event.volume = order_event.totalVolume
                trade_event.tradeTime = self.datetime.strftime('%H:%M:%S')
                self.tradeDict[tradeID] = trade_event
                order_event.tradedVolume = order.totalVolume
                orderv.status = ATATUS_ALLTRADED
                if orderID in self.workingLimitOrderDict:
                    def self.workingLimitOrderDict[orderID]

    def crossStopOrder(self):
        pass
    
    def updateDailyClose(self, datetime, close):
        pass

    #################################################
    def sendOrder(self, symbol, orderType, price, volume):
        self.limitOrderCount += 1
        orderID = str(self.limitOrderCount)
        order_event = OrderEvent()
        order_event.orderID = orderID
        order_event.symbol = symbol
        order_event.price = self.roundToPriceTick(price)
        order_event.totalVolume = volume
        order_event.orderTime = self.datetime.strftime('%H:%M:%S')

        if orderType == CTAORDER_BUY:
            order_event.direction = DIRECTION_LONG
            order_event.offset = OFFSET_OPEN
        elif orderType == CTAORDER_SELL:
            order_event.direction = DIRECTION_SHORT
            order_event.offset = OFFSET_CLOSE
        elif orderType == CTAORDER_SHORT:
            order_event.direction = DIRECTION_SHORT
            order_event.offset = OFFSET_OPEN
        elif orderType == CTAORDER_COVER:
            order_event.direction = DIRECTION_SHORT
            order_event.offset = OFFSET_CLOSE
        
        self.workingLimitOrderDict[orderID] = order_event
        self.limitOrderDict[orderID] = order_event
        return [orderID]
    
    def sendStopOrder(self, symbol, orderType, price, volume, strategyClass):
        self.stopOrderCount += 1
        orderID = str(self.stopOrderCount)
        order_event = StopOrderEvent()
        order_event.symbol = symbol
        order_event.price = self.roundToPriceTick(price)
        order_event.totalVolume = volume
        order_event.strategy = strategyClass
        order_event.status = STOPORDER_WAITING
        order_event.stopOrderID = orderID

        if orderType == CTAORDER_BUY:
            order_event.direction = DIRECTION_LONG
            order_event.offset = OFFSET_OPEN
        elif orderType == CTAORDER_SELL:
            order_event.direction = DIRECTION_SHORT
            order_event.offset = OFFSET_CLOSE
        elif orderType == CTAORDER_SHORT:
            order_event.direction = DIRECTION_SHORT
            order_event.offset = OFFSET_OPEN
        elif orderType == CTAORDER_COVER:
            order_event.direction = DIRECTION_SHORT
            order_event.offset = OFFSET_CLOSE
        
        self.workingStopOrderDict[orderID] = order_event
        self.stopOrderDict[orderID] = order_event
        self.strategyClass.fromStopOrderToTrade(order_event)
        return [orderID]
    
    def cancelOrder(self, orderID):
        if orderID in self.workingLimitOrderDict:
            order_event = workingLimitOrderDict[orderID]
            order_event.status = STATUS_CANCELLED
            order_event.cancelTime = self.datetimee.strftime('%H:%M:%S')
            del self.workingLimitOrderDict[orderID]
    #################################################   
    def getInitData(self):
        return self.initData
    
    def writeLog(self, content):
        self.logList.append(str(self.datetime) + ' ' + content)
    
    def cancelAllOrder(self):
        for orderID in self.workingLimitOrderDict.keys():
            self.cancelOrder(orderID)
        for stopOrderID in self.workingStopOrderDict.keys():
            self.cancleStopOrder(stopOrderID)

    #################################################
    def showBacktestingResults(self):
        if self.mode = self.BAR_MODE:
            latestDataEvent = self.latestBar
        else:
            latestDataEvent = self.latestTick
        performanceClass = Performance(self.mode, self.capital, self.rate, self.slippage, self.size, latestDataEvent)

        d = performanceClass.getPerformanceFromTrades(self.tradeDict)
        performanceClass.outputPerformance(d)
        performanceClass.plotPerformance(d)
        self.output('*' * 30)

        
    