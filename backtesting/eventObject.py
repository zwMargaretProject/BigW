from constant import EMPTY_STRING, EMPTY_FLOAT, EMPTY_INT
from eventType import *

class Event(object):
    def __init__(self):
        self.rawData = None
    

class TickEvent(Event):
    def __init__(self):
        super().__init__()

        self.eventType = EVENT_TICK

        self.symbol = EMPTY_STRING
        self.exchange = EMPTY_STRING
        self.symbolExchange = EMPTY_STRING

        self.period = EMPTY_FLOAT

        self.lastPrice = EMPTY_FLOAT
        self.lastVolume = EMPTY_INT
        self.volume = EMPTY_INT
        self.date = EMPTY_STRING
        self.time = EMPTY_STRING
        self.datetime = None
        
        self.openPrice = EMPTY_FLOAT
        self.lastPrice = EMPTY_FLOAT
        self.lowPrice = EMPTY_FLOAT
        self.preClosePrice = EMPTY_FLOAT

        self.upperLimit = EMPTY_FLOAT
        self.lowerLimit = EMPTY_FLOAT

        self.bidPrice1 = EMPTY_FLOAT
        self.bidPrice2 = EMPTY_FLOAT
        self.bidPrice3 = EMPTY_FLOAT
        self.bidPrice4 = EMPTY_FLOAT
        self.bidPrice5 = EMPTY_FLOAT

        self.askPrice1 = EMPTY_FLOAT
        self.askPrice2 = EMPTY_FLOAT
        self.askPrice3 = EMPTY_FLOAT
        self.askPrice4 = EMPTY_FLOAT
        self.askPrice5 = EMPTY_FLOAT

        self.bidVolume1 = EMPTY_INT
        self.bidVolume2 = EMPTY_INT
        self.bidVolume3 = EMPTY_INT
        self.bidVolume4 = EMPTY_INT
        self.bidVolume5 = EMPTY_INT

        self.askVolume1 = EMPTY_INT
        self.askVolume2 = EMPTY_INT
        self.askVolume3 = EMPTY_INT
        self.askVolume4 = EMPTY_INT
        self.askVolume5 = EMPTY_INT



def BarEvent(Event):
    def __init__(self):
        super().__init__()

        self.eventType = EVENT_BAR

        self.symbol = EMPTY_STRING
        self.exchange = EMPTY_STRING
        self.symbolExchange = EMPTY_STRING

        self.period = EMPTY_FLOAT

        self.open = EMPTY_FLOAT
        self.high = EMPTY_FLOAT
        self.low = EMPTY_FLOAT
        self.close = EMPTY_FLOAT

        self.date = EMPTY_STRING
        self.time = EMPTY_STRING
        self.datetime = None

        self.volume = EMPTY_INT
        self.openInterest = EMPTY_INT


class TradeEvent(Event):
    def __init__(self):
        super().__init__()

        self.eventType = EVENT_TRADE

        self.symbol = EVENT_STRING
        self.exchange = EMPTY_STRING
        self.symbolExchange = EMPTY_STRING

        self.tradeID = EMPTY_STRING
        self.orderID = EMPTY_STRING

        self.direction = EMPTY_UNICODE
        self.offset = EMPTY_UNICODE
        self.price = EMPTY_FLOAT
        self.volume = EMPTY_INT
        self.tradeTime = EMPTY_STRING


class OrderEvent(Event):
    def __init__(self):
        super().__init__()

        self.eventType = EVENT_ORDER

        self.symbol = EMPTY_STRING
        self.exchange = EMPTY_STRING
        self.symbolExchange = EMPTY_STRING

        self.orderID = EMPTY_STRING

        self.direction = EMPTY_UNICODE
        self.offset = EMPTY_UNICODE
        self.price = EMPTY_FLOAT
        self.totalVolume = EMPTY_INT
        self.tradedVolume = EMPTY_INT
        self.status = EMPTY_UNICODE

        self.orderTime = EMPTY_STRING
        self.cancleTime = EMPTY_STRING


class PositionEvent(Event):
    def __init__(self):
        super().__init__()

        self.eventType = EVENT_POSITION

        self.symbol = EMPTY_STRING
        self.exchange = EMPTY_STRING
        self.symbolExchanbe = EMPTY_STRING

        self.direction = EMPTY_UNICODE
        self.position = EMPTY_INT
        self.frozen = EMPTY_INT
        self.price = EMPTY_FLOAT
        self.positionName = EMPTY_STRING

        self.prevPosition = EMPTY_INT
        self.positionProfit = EMPTY_FLOAT


class AccountEvent(Event):
    def __init__(self):
        super().__init__()

        self.eventType = EVENT_ACCOUNT

        self.accountID = EMPTY_FLOAT
        self.balance = EMPTY_FLOAT
        self.available = EMPTY_FLOAT
        self.commission = EMPTY_FLOAT
        self.margin = EMPTY_FLOAT
        self.closeProfit = EMPTY_FLOAT
        self.positionProfit = EMPTY_FLOAT


class ErrorEvent(Event):
    def __init__(self):
        super().__init__()

        self.eventType = EVENT_ERROR

        self.errorID = EMPTY_STRING
        self.errorMsg = EMPTY_UNICODE
        self.additionalInfo = EMPTY_UNICODE

        self.errorTime = time.strftime('%X', time.localtime())


class LogEvent(Event):
    def __init__(self):
        super().__init__()

        self.eventType = EVENT_LOG

        self.logTime = time.strftime('%X', time.localtime())
        self.logContent = EMPTY_UNICODE






         
        