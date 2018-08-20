# core
import os
import datetime

# installed
import quandl
import pandas as pd
import numpy as np
from pytz import timezone
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor

# custom
from utils import get_home_dir
import dl_quandl_EOD as dq

stocks = dq.load_stocks()


DEFAULT_STORAGE = '/home/nate/eod_data/'
# get todays date for checking if files up-to-date
MTN = timezone('America/Denver')
TODAY = datetime.datetime.now(MTN)
WEEKDAY = TODAY.weekday()
HOUR = TODAY.hour

HOME_DIR = get_home_dir()

Q_KEY = os.environ.get('quandl_api')

quandl.ApiConfig.api_key = Q_KEY

spy_vix = {}
closes = {}
dates = {}
for i in range(1, 10):
    print(i)
    spy_vix[i] = quandl.get("CHRIS/CBOE_VX" + str(i))
    # spy_vix[i].to_csv()
    closes[i] = spy_vix[i]['Close']
    dates[i] = spy_vix[i].index

for i in range(1, 10):
    print(i)
    print(dates[i][0])

# inner join on dfs

spy_vix_df_7 = pd.DataFrame(spy_vix[1]['Close'].copy())
for i in range(2, 8):
    spy_vix_df_7 = spy_vix_df_7.merge(pd.DataFrame(spy_vix[i]['Close']), left_index=True, right_index=True)

spy_vix_df_7.columns = range(1, 8)

# contango/backwardation plots
# for i, r in spy_vix_df_7.iloc[-100:].iterrows():
#     plt.plot(range(1, 8), r)
#
# plt.show()

# set to negative so contango is positive, backwardation is negative
spy_vix_df_7['m1_2_log_ratio'] = - np.log(spy_vix_df_7[1] / spy_vix_df_7[2]) / 30 * 365
# percent contango as referenced here: http://vixcentral.com/
spy_vix_df_7['m1_2_pct_ratio'] = (spy_vix_df_7[2] - spy_vix_df_7[1]) / spy_vix_df_7[1]

spy_vix = spy_vix_df_7.copy()
spy_vix_ = spy_vix.merge(stocks['SPY'], left_index=True, right_index=True)
# uvxy limits data to 2012+
# spy_vix_ = spy_vix_.merge(stocks['UVXY'], left_index=True, right_index=True, suffixes=('_SPY', '_UVXY'))

# get pct price change in 5 days
# the idea is have information on a day.  Buy at open next day, hold for 5 days, sell on 6th day
# this gets % price change from next day to 5 days after that

spy_vix_['5d_future_pct_change_SPY'] = spy_vix_['Adj_Open'].shift(-1).pct_change(5).shift(-5)
spy_vix_['1d_future_pct_change_SPY'] = spy_vix_['Adj_Open'].shift(-1).pct_change(1).shift(-1)
# this is for loading SPY and UVXY
# spy_vix_['5d_future_pct_change_SPY'] = spy_vix_['Adj_Open_SPY'].shift(-1).pct_change(5).shift(-5)
# spy_vix_['5d_future_pct_change_UVXY'] = spy_vix_['Adj_Open_UVXY'].shift(-1).pct_change(5).shift(-5)
# spy_vix_['1d_future_pct_change_SPY'] = spy_vix_['Adj_Open_SPY'].shift(-1).pct_change(1).shift(-1)
# spy_vix_['1d_future_pct_change_UVXY'] = spy_vix_['Adj_Open_UVXY'].shift(-1).pct_change(1).shift(-1)
spy_vix_.corr()
spy_vix_.iloc[-300:].corr()

# bunch of plots
# plt.scatter(spy_vix_[1], spy_vix_['5d_future_pct_change_SPY'])
# plt.show()
#
# plt.scatter(spy_vix_[7], spy_vix_['5d_future_pct_change_SPY'])
# plt.show()
#
# plt.scatter(spy_vix_['m1_2_log_ratio'], spy_vix_['5d_future_pct_change_SPY'])
# plt.show()
#
# # line plot of contango percent and price
# spy_vix_['Adj_Open'].plot()
# spy_vix_['m1_2_log_ratio'].plot(secondary_y=True)
# plt.show()

# load in VIX data, make target as close price 2 days in future
vix = pd.read_csv('/home/nate/github/scrape_stocks/data/VIX/vixcurrent.csv', skiprows=1, parse_dates=['Date'], index_col='Date', infer_datetime_format=True)
vix['2d_future_close'] = vix['VIX Close'].shift(-2)
# so if you were to buy it the next day at open and sell at close 2 days later, this is about the return
vix['target'] = vix['VIX Close'].pct_change(2).shift(-2)

# look at correlation between vix and vxx price
vix_vxx = stocks['VXX'].merge(vix, left_index=True, right_index=True)

# not that correlated -- 74% pearson
vix_vxx.plot.scatter('VIX Close', 'Close')
plt.show()

vix_vxx['VIX Close'].plot()
vix_vxx['Close'].plot(secondary_y=True)
plt.show()




# features are current vix futures, 5, 10, 15, 20, 50, 200d EMA, RSI
#
