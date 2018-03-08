# core
import time
import sys
from concurrent.futures import ProcessPoolExecutor

# custom
sys.path.append('../code')
import dl_quandl_EOD as dq
import data_processing as dp
import short_squeeze_eda as sse
from utils import get_home_dir
HOME_DIR = get_home_dir(repo_name='scrape_stocks')

# installed
import numpy as np
import pandas as pd
from tqdm import tqdm

# plotting
import matplotlib.pyplot as plt
%matplotlib inline
from plotly.offline import init_notebook_mode, iplot
from plotly.graph_objs import Scatter, Scattergl, Figure, Layout

def calc_vol(st, mean_vol):
    """
    takes dataframe of stock data (st) and calculates tp, 50d-mva, and volatility
    also takes dictionary (mean_vol) as arg
    """
    st['typical_price'] = st[['Adj_High', 'Adj_Low', 'Adj_Close']].mean(axis=1)
    st['50d_mva'] = talib.SMA(st['typical_price'].values, timeperiod=50)
    st['volatility'] = ((st['typical_price'] - -st['50d_mva'])/st['typical_price']).rolling(50).std()
    mean_vol[t] = np.mean(st['volatility'])


stocks = dq.load_stocks()
tickers = sorted(stocks.keys())


# get stocks that are still trading and have larger volumes
vols = []
latest_tickers = []
for t in tickers:
    if latest_date in stocks[t].index:
        vol = np.mean(stocks[t].iloc[-100:]['Adj_Volume'] * stocks[t].iloc[-100:]['Adj_Close'])
        if vol > 1e8:  # 10 million or greater per day
            vols.append(vol)
            latest_tickers.append(t)


# need to multithread...
# calculate volatility from 50D MVA
mean_vol = {}
for t in tqdm(latest_tickers):
    stocks[t]['typical_price'] = stocks[t][['Adj_High', 'Adj_Low', 'Adj_Close']].mean(axis=1)
    stocks[t]['50d_mva'] = talib.SMA(stocks[t]['typical_price'].values, timeperiod=50)
    stocks[t]['volatility_ewma'] = ((stocks[t]['typical_price'] - stocks[t]['50d_mva'])/stocks[t]['typical_price']).ewm(50).std()
    stocks[t]['volatility'] = ((stocks[t]['typical_price'] - stocks[t]['50d_mva'])/stocks[t]['typical_price']).rolling(50).std()
    mean_vol[t] = np.mean(stocks[t]['volatility'])

mean_vols = [mean_vol[t] for t in latest_tickers]

vol_idx = np.argsort(mean_vols)[::-1]
sorted_vols = np.array(mean_vols)[vol_idx]
best_and_old = []
best_old_vols = []
for i, s in enumerate(np.array(latest_tickers)[vol_idx]):
    if stocks[s].index.min() < pd.to_datetime('2000-01-01'):
        best_and_old.append(s)
        best_old_vols.append(sorted_vols[i])

# make dataframe with rolling vol, vol, and mean_vol
# raw data here -- old and good tickers are below further
vol_dict = {'ticker': latest_tickers,
           'mean_vol': mean_vols,
           'ewm_vols': ewm_vols}
vol_df = pd.DataFrame(vol_dict)
# save for later
vol_df.to_csv('mean_volatilities.csv')


# make dataframe with rolling vol, vol, and mean_vol
vol_dict = {'ticker': best_and_old,
           'mean_vol': best_old_vols,
           'ewm_vols': best_old_ewm}
vol_df = pd.DataFrame(vol_dict)
# save for later
vol_df.to_csv('mean_volatilities_best_and_old.csv')

# export list of top volatile stocks by the EMA, plus indices
sorted_old = vol_df_best_old.sort_values(by='ewm_vols', ascending=False)
sorted_old.to_csv('old_good_sorted_by_ewm.csv')
top_stocks = sorted_old.loc[:100, 'ticker'].tolist() + ['SPY', 'UPRO', 'QQQ', 'TQQQ', 'DIA', 'UBT']
