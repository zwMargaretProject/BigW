from eventType import (TickEvent, BarEvent)

class BarGenerator(object):
	'''
	1. Merge tick data to bar data;
	2. Merge 1-minute bar data to x-minute bar dat.
	'''

	def __init__(self, fromBarToOrder_func, x, x_fromBarToOrder_func=None):
		self.now_bar_event = None
		self.fromBarToOrder_func = fromBarToOrder_func
		self.now_x_bar_event = None
		self.x = x
		self.x_fromBarToOrder_func = x_fromBarToOrder_func

		self.lastTick = None

	def updateTick(self, tick_event):
		newMinute = False
		if self.now_bar_event is None:
			self.now_bar_event = BarEvent()
			newMinute = True
		elif self.now_bar_event.datetime.minute != tick_event.datetime.minute:
			self.now_bar_event.datetime = self.now_bar_event.datetime.replace(second=0, microsecond=0)
			self.now_bar_event.date = self.now_bar_event.strftime('%Y%m%d')
			self.now_bar_event.time = self.now_bar_event.datetime.strftime('%H:%M:%S.%f')
			self.fromBarToOrder_func(self.now_bar_event)
			self.now_bar_event = BarEvent()
			newMinute = True

		fixedPrice = tick_event.lastPrice
		if newMinute is True:
			self.now_bar_event.symbol = tick_event.symbol
			self.now_bar_event.open = fixedPrice
			self.now_bar_event.high = fixedPrice
			self.now_bar_event.low = fixedPrice
		else:
			self.now_bar_event.high = max(self.now_bar_event.high, fixedPrice)
			self.now_bar_event.low = min(self.now_bar_event.low, fixedPrice)
		self.now_bar_event.close = fixedPrice
		self.now_bar_event.datetime = tick_event.datetime
		self.now_bar_event.openInterest = tick_event.openInterest

		if self.lastTick is not None:
			self.now_bar_event.volume += (tick_event.volume -  self.lastTick.volume)
		self.lastTick = tick_event


	def updateBar(self, bar_event):
		if self.x_bar_event is None:
			self.x_bar_event = BarEvent()
			self.x_bar_event.symbol = bar_event.symbol
			self.x_bar_event.open = bar_event.open
			self.x_bar_event.high = bar_event.high
			self.x_bar_event.low = bar_event.low
			self.x_bar_event.datetime = bar_event.datetime
		else:
			self.x_bar_event.high = max(self.x_bar_event.high, bar_event.high)
			self.x_bar_event.low = min(self.x_bar_event.low, bar_event.low)
		self.x_bar_event.close = bar_event.close
		self.x_bar_event.openInterest = bar_event.openInterest
		self.x_bar_event.volume += int(bar_event.volume)

		if not (bar_event.datetime.minute + 1) % self.x:
			self.x_bar_event.datetime = self.x_bar_event.datetime.replace(second=0, microsecond=0)
			self.x_bar_event.date = self.x_bar_event.datetime.strftime('%Y%m%d')
			self.x_bar_event.time = self.x_bar_event.datetime.strftime('%H:%M:%S.%f')
			self.x_fromBarToOrder_func(self.x_bar_event)
			self.x_bar_event = None

