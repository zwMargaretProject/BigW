
import math
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['font.family'] = 'serif'
from bsm_option_valuation import bsm_call_value

S0 = 100.0
K = 100.0
T = 1.0
r = 0.05
sigma = 0.2

def crr_option_value(S0, K, T, r, sigma, otype, M=4):
    dt = T / M
    df = math.exp(-r * dt)

    u = math.exp(sigma * math.sqrt(dt))
    d = 1 / u
    q = (math.exp(r * dt) - d) / (u - d)