import datetime
import numpy as np
import pandas as pd
from abc import ABCMeta, abstractmethod
import pandas as pd


class QuickStrategy(object):
	__metaclass__ = ABCMeta

	@abstractmethod
	def generate_signals(self):
		pass

		

