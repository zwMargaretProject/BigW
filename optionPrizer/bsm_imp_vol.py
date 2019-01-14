
from math import log, sqrt, exp
from scipy import stats
from scipy.optimize import fsolve

class EuropeanOption(object):
    def __init__(self, S0, K, t, M, r, sigma):
        try:
            self.S0 = float(S0)
            self.K = float(K)
            self.t = float(t)
            self.M = float(M)
            self.r = float(r)
            self.sigma = float(sigma)
        except ValueError:
            print("Please input valid numbers or strings.")
            raise
    
    def update_ttm(self):
        if self.t > self.M:
            raise ValueError("Pricing date is later than maturity.")
        self.T = (self.M - self.t).days / 365.0
  
    def d1(self):
        d1 = ((log(self.S0 / self.K) + (self.r + 0.5 * self.sigma ** 2) * self.T) / (self.sigma * sqrt(self.T)))
        return d1
    
    def d1_d2(self):
        d1 = self.d1()
        d2 = self.d1 - (self.sigma * sqrt(self.T))
        return d1, d2
    
    @abstracmethod
    def option_value(self):
        pass
    
    @abstractmethod
    def delta(self):
        pass
    
    @property
    def vega(self):
        self.update_ttm()
        d1 = self.d1()
        vega = self.S0 * stats.norm.pdf(d1, 0.0, 1.0) * sqrt(self.T)
        return vega

    @abstractmethod
    def imp_vol(self, option_value, sigma_init=0.2):
        pass



class CallOption(EuropeanOption):
    def __init__(self, S0, K, t, M, r, sigma):
        super().__init__(self, S0, K, t, M, r, sigma)
    
    @property
    def option_value(self):
        d1, d2 = self.d1_d2()
        value = (self.S0 * stats.norm.cdf(d1, 0.0, 1.0) - self.K * exp(-self.r * self.T) * stats.norm.cdf(d2, 0.0, 1.0))
        return value

    @property
    def delta(self):
        delta = stats.norm.cdf(d1, 0.0, 1.0)
        return delta
    
    @property
    def imp_vol(self, market_value, sigma_init=0.2):
        option = CallOption(self.S0, self.K, self.t, self.M, self.r, sigma_init)
        option.update_ttm()
        def difference(sigma):
            option.sigma = simga
            return option.value() - market_value
        imp_vol = fsove(difference, sigma_init)[0]
        return imp_vol
        
class PutOption(EuropeanOption):
    def __init__(self, S0, K, t, M, r, sigma):
        super().__init__(self, S0, K, t, M, r, sigma)
    
    @property
    def option_value(self):
        d1, d2 = self.d1_d2()
        value = (-self.S0 * stats.norm.cdf(-d1, 0.0, 1.0) + self.K * exp(-self.r * self.T) * stats.norm.cdf(-d2, 0.0, 1.0))
        return value

    @property
    def delta(self):
        delta = stats.norm.cdf(d1, 0.0, 1.0) - 1
        return delta
    
    @property
    def imp_vol(self, market_value, sigma_init=0.2):
        option = PutOption(self.S0, self.K, self.t, self.M, self.r, sigma_init)
        option.update_ttm()
        def difference(sigma):
            option.sigma = simga
            return option.value() - market_value
        imp_vol = fsove(difference, sigma_init)[0]
        return imp_vol
        
        



    

    

