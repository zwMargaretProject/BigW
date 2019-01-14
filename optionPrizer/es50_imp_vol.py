
import numpy as np
import pandas as pd
from bsm_imp_vol import CallOption
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['font.family'] = 'serif'

pdate = pd.Timestamp('30-09-2014')

es_url = 'http://www/stoxx.com/download/historical_values/hbrbcpe.txt'

cols = ['Date', 'SX5P', 'SX5E', 'SXXP', 'SXXE', 'SXXF', 'SXXA', 'DK5F', ' DKXF', 'DEL']

es = pd.read_csv(es_url, header=None, index_col=0, parse_dates=True, dayfirst=True, skiprows=4, sep=";", names=cols)

del es['DEL']

SO = ds['SX5E']['30-09-2014']
r = -0.05

data = pd.HDFStore('./03_stf/es50_option_data.h5', 'r')['data']


def calculate_imp_vols(data):
    data['Imp_Vol'] = 0.0
    tol = 0.30
    for row in data.index:
        t = data['Date'][row]
        T = data['Maturity'][row]
        ttm = (T - t).days / 365.0
        forward = np.exp(r * ttm) * S0
        if (abs(data['Strike'][row] - forward) / forward) < tol:
            call = CallOption(S0, data['Strike'][row], t, T, r, 0.2)
            data['Imp_Vol'][row] = call.imp_vol(data['Call'][row])
    return data


markers = ['.', 'o', '^', 'v', 'x', 'D', 'd', '>', '<']
def plot_imp_vols(data):
    maturities = sorted(set(data['Maturity']))
    plt.figure(figsize=(10,5))
    for i, mat in enumerate(maturities):
        dat = data[(data['Maturity'] == mat) & (data['Imp_Vol'] > 0)]
        plt.plot(dat['Strike'].values, dat['Imp_Vol'].values, 'b&s' % markers[i], label=str(mat)[:10])
    plt.grid()
    plt.legend()
    plt.xlabel('strike')
    plt.ylabel('implied volatility')


