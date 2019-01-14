
import math
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['font.family'] = 'serif'
from scipy.integrate import quad

class BlackSholesMertonPrizer(object):
	def __init__(self):
		pass

	def dN(self, x):
	    return math.exp(-0.5 * x ** 2) / math.sqrt(2 * math.pi)

	def N(self, d):
	    return quad(lambda x: self.dN(x), -20, d, limit=50)[0]

	def d1_function(self, St, K, t, T, r, sigma):
	    d1 = (math.log(St / K) + (r + 0.5 * sigma ** 2) * (T - t)) / (sigma * math.sqrt(T - t))
	    return d1

	def call_value(self, St, K, t, T, r, sigma):
	    d1 = self.d1_function(St, K, t, T, r, sigma)
	    d2 = d1 - (sigma * math.sqrt(T - t))
	    call_value = St * self.N(d1) - math.exp(-r * (T - t)) * K * self.N(d2)
	    return call_value

	def bsm_put_value(self, St, K, t, T, r, sigma):
	    d1 = self.d1_function(St, K, t, T, r, sigma)
	    d2= d1 - (sigma * math.sqrt(T - t))
	    put_value = - St * self.N(-d1) + math.exp(-r * (T - t)) * K * self.N(-d2)
	    return put_value

	def plot_values(function):
	    plt.figure(figsize=(10, 8.3))
	    points = 100

	    St = 100.0
	    K = 100.0
	    t = 0.0
	    T = 1.0
	    r = 0.05
	    sigma = 0.2

	    plt.subplot(221)
	    klist = np.linspace(80, 120, points)
	    vlist = [function(St, K, t, T, r, sigma) for K in klist]
	    plt.plot(klist, vlist)
	    plt.grid()
	    plt.xlabel('strike $K$')
	    plt.ylabel('present value')

	    plt.subplot(222)
	    tlist = np.linspace(0.0001, 1, points)
	    vlist =  [function(St, K, t, T, r, sigma) for T in tlist]
	    plt.plot(tlist, vlist)
	    plt.grid()
	    plt.xlabel('maturity $T$')

	    plt.subplot(223)
	    rlist = np.linspace(0, 0.1, points)
	    vlist =  [function(St, K, t, T, r, sigma) for  r in rlist]
	    plt.plot(rlist, vlist)
	    plt.grid()
	    plt.xlabel('short rate $r$')
	    plt.ylabel('present value')
	    plt.axis('tight')

	    plt.subplot(224)
	    slist = np.linspace(0.01, 0.5, points)
	    vlist =  [function(St, K, t, T, r, sigma) for  sigma in slist]
	    plt.plot(slist, vlist)
	    plt.grid()
	    plt.xlabel('volatility $sigma$')
	    plt.ylabel('present value')
	    plt.tight_layout()

	    


	    

