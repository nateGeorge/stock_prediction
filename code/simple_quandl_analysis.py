# installed
import pandas as pd
import numpy as np
import talib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import ParameterGrid
import matplotlib.pyplot as plt

# custom
import data_processing as dp


def load_stocks_calculate_short_corr():
    dfs, sh_int, fin_sh = dp.load_stocks(stocks=None)

    latest_stocks = []
    all_sh_stocks = []
    all_sh_stocks_full = []
    latest_date = sh_int['SPY'].index[-1]
    for s in sh_int.keys():
        if sh_int[s].shape[0] == 0:
            print(s, 'is empty')
            continue

        if latest_date != sh_int[s].index[-1]:
            print(s, 'is old')
            continue

        print(s)
        df = sh_int[s].copy()
        df['5d_price_change'] = df['Adj_Close'].pct_change(5).shift(-5)
        df['10d_price_change'] = df['Adj_Close'].pct_change(10).shift(-10)
        df['20d_price_change'] = df['Adj_Close'].pct_change(20).shift(-20)
        df['40d_price_change'] = df['Adj_Close'].pct_change(40).shift(-40)
        df['ticker'] = s

        # create short-close correlations -- need to deal with -1s
        # if short % is all -1 or 0, won't work.  if less than 20 samples, rolling corr with 20 period window won't work
        # also broke on DF with 22 samples
        if df['Short_%_of_Float'].mean() in [-1, 0] or df.shape[0] < 30:
            df['Short_%_of_Float_10d_EMA'] = -np.inf
            df['Adj_Close_10d_EMA'] = talib.EMA(df['Adj_Close'].values, timeperiod=10)
            df['short_close_corr_10d_EMA'] = -np.inf
            df['short_close_corr_rocr_20d'] = -np.inf
            df['short_%_rocr_20d'] = -np.inf
        else:
            df['Short_%_of_Float_10d_EMA'] = talib.EMA(df['Short_%_of_Float'].values, timeperiod=10)
            df['Adj_Close_10d_EMA'] = talib.EMA(df['Adj_Close'].values, timeperiod=10)

            # essentially, we want to take an arbitrary number of points, calculate correlation, and find where the correlations are largest
            # take 10 points at a time and get correlations first, then take parts that have largest correlations, and keep expanding by 5 points at a time until correlation decreases
            corr = df[['Short_%_of_Float_10d_EMA', 'Adj_Close_10d_EMA']].rolling(window=20).corr()
            df['short_close_corr_10d_EMA'] = corr.unstack(1)['Short_%_of_Float_10d_EMA']['Adj_Close_10d_EMA']
            df['short_close_corr_10d_EMA'].replace(np.inf, 1, inplace=True)
            df['short_close_corr_10d_EMA'].replace(-np.inf, -1, inplace=True)
            df['short_close_corr_10d_EMA'].clip(lower=-1, upper=1, inplace=True)

            # WARNING: things with small (< 1%) Short % of float will result in huge rocr...maybe do something about this
            df['short_close_corr_rocr_20d'] = talib.ROCR100(df['short_close_corr_10d_EMA'].values, timeperiod=20)
            df['short_%_rocr_20d'] = talib.ROCR100(df['Short_%_of_Float_10d_EMA'].values, timeperiod=20)

            # auto-detect long stretches of negative and positive correlation
            thresh = 0.7
            rolling = df['short_close_corr_10d_EMA'].rolling(window=20).min()
            df['Short_%_positive_corr_detection'] = rolling > thresh
            df['Short_%_positive_corr_detection'] = df['Short_%_positive_corr_detection'].astype('int16')
            # sh_int[ticker]['Short_%_positive_corr_detection'].plot()
            # plt.show()

            df['Short_%_negative_corr_detection'] = rolling < -thresh
            df['Short_%_negative_corr_detection'] = df['Short_%_negative_corr_detection'].astype('int16')


        latest_stocks.append(df.iloc[-1])
        all_sh_stocks_full.append(df)
        all_sh_stocks.append(df.dropna())

    latest_stocks_df = pd.concat(latest_stocks, axis=1).T
    latest_stocks_df.set_index('ticker', inplace=True)

    all_sh_stocks_df = pd.concat(all_sh_stocks)
    all_sh_stocks_df['market_cap'] = all_sh_stocks_df['Shares_Outstanding'] * all_sh_stocks_df['Adj_Close']
    all_sh_stocks_full_df = pd.concat(all_sh_stocks_full)

    return all_sh_stocks_df, all_sh_stocks_full_df, latest_stocks_df, sh_int


all_sh_stocks_df, all_sh_stocks_full_df, latest_stocks_df, sh_int = load_stocks_calculate_short_corr()

def make_larger_shorts(all_sh_stocks_df):
    # plot short close rocr 20d to see if trend from TDOC holds overall -- under 80% ROCR100 means high gains
    # no overall trend, but for individual stocks there tends to be a trend
    shorts = all_sh_stocks_df[all_sh_stocks_df['short_%_rocr_20d'] != -np.inf]
    shorts_nona = shorts.dropna()

    ticker_groups = shorts_nona[['ticker', 'Short_%_of_Float', 'Days_to_Cover', 'rocp_cl', 'short_%_rocr_20d', 'short_close_corr_rocr_20d']].groupby('ticker').mean()
    larger_shorts = ticker_groups[ticker_groups['Short_%_of_Float'] > 10]
    short_stocks = larger_shorts.index

    return ticker_groups, larger_shorts, short_stocks


def short_squeeze_analysis(all_sh_stocks_df, ticker='TDOC'):
    # also look at EMA
    df = all_sh_stocks_df[all_sh_stocks_df['ticker'] == ticker]

    # I think all this is already calcuclated in load_stocks_calculate_short_corr
    # df['Short_%_of_Float_10d_EMA'] = talib.EMA(sh_int[ticker]['Short_%_of_Float'].values, timeperiod=10)
    # df['Adj_Close_10d_EMA'] = talib.EMA(df['Adj_Close'].values, timeperiod=10)
    df[['Short_%_of_Float', 'Short_%_of_Float_10d_EMA']].plot()

    # look for sections where short % mva and price are (ideally, inversely) correlated
    # make gradient of color where oldest -> newest goes from dark to light
    cm = plt.get_cmap('copper')
    num_colors = df.shape[0]
    colors = []
    for i in range(num_colors):
        colors.append(cm(i / num_colors))  # color will now be an RGBA tuple

    df.plot.scatter(x='Short_%_of_Float_10d_EMA', y='Adj_Close', color=colors)
    plt.show()

    # essentially, we want to take an arbitrary number of points, calculate correlation, and find where the correlations are largest
    # take 10 points at a time and get correlations first, then take parts that have largest correlations, and keep expanding by 5 points at a time until correlation decreases
    # already being done in preprocessing now
    # corr = df[['Short_%_of_Float_10d_EMA', 'Adj_Close_10d_EMA']].rolling(window=20).corr()
    # df['short_close_corr_10d_EMA'] = corr.unstack(1)['Short_%_of_Float_10d_EMA']['Adj_Close_10d_EMA']
    # df['short_close_corr_10d_EMA'].replace(np.inf, 1, inplace=True)
    # df['short_close_corr_10d_EMA'].replace(-np.inf, -1, inplace=True)
    # df['short_close_corr_10d_EMA'].clip(lower=-1, upper=1, inplace=True)

    # rows, columns
    # f, ax = plt.subplots(nrows=2, ncols=1, sharex=True)
    # sh_int[ticker]['short_close_corr_10d_EMA'].plot(ax=ax[0])
    # sh_int[ticker][['Adj_Close_10d_EMA', 'Short_%_of_Float_10d_EMA']].plot(ax=ax[1])
    # plt.show()

    # auto-detect long stretches of negative and positive correlation
    # now calculated in load_stocks_calculate_short_corr function
    # thresh = 0.8
    # rolling = df['short_close_corr_10d_EMA'].rolling(window=20).min()
    # df['Short_%_positive_corr_detection'] = rolling > thresh
    # df['Short_%_positive_corr_detection'] = df['Short_%_positive_corr_detection'].astype('int16')
    # # sh_int[ticker]['Short_%_positive_corr_detection'].plot()
    # # plt.show()
    #
    # df['Short_%_negative_corr_detection'] = rolling < -thresh
    # df['Short_%_negative_corr_detection'] = df['Short_%_negative_corr_detection'].astype('int16')
    df[['short_close_corr_10d_EMA', 'Short_%_negative_corr_detection', 'Adj_Close', 'Short_%_of_Float_10d_EMA', 'Days_to_Cover', 'short_close_corr_rocr_20d']].plot(subplots=True)
    plt.show()


ticker_groups, larger_shorts, short_stocks = make_larger_shorts(all_sh_stocks_df)

# still 2 million datapoints, so going to sample down...
sample = shorts_nona.sample(20000)
ticker = 'AAPL'
sample = shorts_nona[shorts_nona['ticker'] == ticker]
plt.scatter(sample['short_%_rocr_20d'], sample['20d_price_change'])
plt.xlim(0, 200)
plt.ylim(-2, 2)
plt.show()

shorts_nona_ssr = shorts_nona[shorts_nona['Short_Squeeze_Ranking'] != -1]
shorts_nona_ssr[['Short_Squeeze_Ranking', '20d_price_change']].corr()
shorts_nona_ssr[['Performance_(52-wk)', '20d_price_change']].corr()
shorts_nona_ssr[['rsi_cl', '20d_price_change']].corr()
shorts_nona_ssr[['Performance_(52-wk)', '40d_price_change']].corr()
shorts_nona_ssr[['Short_Squeeze_Ranking', '%_from_52-wk_High']].corr()
shorts_nona_ssr[['Short_Squeeze_Ranking',
                'Total_Short_Interest',
                'Days_to_Cover',
                'Short_%_of_Float',
                'Performance_(52-wk)',
                '%_Insider_Ownership',
                '%_Institutional_Ownership',
                '%_from_52-wk_High',
                '%_from_200_day_MA',
                '%_from_50_day_MA',
                'Shares_Float',
                'Avg._Daily_Vol.',
                'Shares_Outstanding',
                '%_Change_Mo/Mo'
                ]].corr()
# most highly correlated with 'performance'.  Definition from site:
"""
This is an indicator of the relative strength of a stock over the past 52 weeks. A strong number indicates strong performance and a low number indicates weak performance.
"""
# days to cover looks to be next
sample = shorts_nona_ssr.sample(200000)
plt.scatter(sample['Short_Squeeze_Ranking'], sample['Performance_(52-wk)'], alpha=0.1)
plt.show()

sample = shorts_nona_ssr.sample(200000)
plt.scatter(sample['Short_Squeeze_Ranking'], sample['Performance_(52-wk)'], alpha=0.1)
plt.show()

plt.scatter(sample['rsi_cl'], sample['20d_price_change'], alpha=0.1)
plt.show()
shorts_nona
# next todo: get correlation between short_%_rocr and the price change for each stock, and find those most correlated
# also screen for stocks like TDOC with exceptional areas
# filter for market caps
# duh...screen for high amounts of short (> 10% or something).  e.g. AEP not correlated, but shorting is low and constant at 1% ish
# factor in days to cover with short % correlation
# get earnings report dates and filter by those with earnings date nearing -- check if earnings is often the cause of a squeeze
shorts_nona['market_cap'] = shorts_nona['Shares_Outstanding'] * shorts['Adj_Close']
shorts_nona['market_cap'].plot.hist(bins=100)#, loglog=True)
plt.xlim([500000, 1000000000])
plt.show()


#0 AEP - no correlation, AER - slight correlation, AERI -- complicated, EXAS - year long short squeeze
#1 AAMC great example of short squeeze - 4 month period
#2 AAOI -- potential
#2 AAXN another great example -- looks like a large margin call (tazers) - days to cover was in 30s
#3 ABAX, similar to AAXN and almost same time -- was a big earnings surprise I think (blood analysis) -- days to cover was in 40s
#4 ABEO - prices keep going up as do shorts? -- could be a short squeeze if earnings improves significantly
#5 any news on the horizon for anything promising?  how to automate this with text analytics?
#6 ABG - weak shorting
#7 ACAD - pretty good short squeeze in 2017
#8 ACHC - long squeeze, actually kind of a pile-on
#9 ACHN - another long squeeze
#10 ACIA - slowly dying long squeeze
# DAVE also an example
ticker = short_stocks[3]
ticker = 'WING'
print(ticker)
print(larger_shorts.loc[ticker])
short_squeeze_analysis(all_sh_stocks_full_df, ticker=ticker)

# detect current short-squeezers -- look for increasing price (rocr positive on close EMA, and rocr negative on Short %), as well as high days to cover
# may not work well -- with all_sh_stocks_df, allows some backtesting -- didn't work well for VHC.  probably need stronger rocr_cl_100
# setup autodetection with these
ticker_groups = all_sh_stocks_full_df[['ticker', 'Short_%_of_Float', 'Days_to_Cover', 'rocp_cl', 'short_%_rocr_20d', 'short_close_corr_rocr_20d']].groupby('ticker').tail(1)
ticker_groups[(ticker_groups['Short_%_of_Float'] > 10) & (ticker_groups['Days_to_Cover'] > 10) & (ticker_groups['rocp_cl'] > .10) & (ticker_groups['short_%_rocr_20d'] < 95)]
ticker_groups[(ticker_groups['Short_%_of_Float'] > 10) & (ticker_groups['Days_to_Cover'] > 10) & (ticker_groups['rocp_cl'] > .05)].sort_values(by='Days_to_Cover')
ticker_groups[(ticker_groups['Short_%_of_Float'] > 10) & (ticker_groups['Days_to_Cover'] > 10) & (ticker_groups['rocp_cl'] > .05)].sort_values(by='rocp_cl')
ticker_groups[(ticker_groups['Short_%_of_Float'] > 10) & (ticker_groups['Days_to_Cover'] > 10) & (ticker_groups['rocp_cl'] > .05)].sort_values(by='short_%_rocr_20d')
ticker_groups[(ticker_groups['Short_%_of_Float'] > 10) & (ticker_groups['Days_to_Cover'] > 10)].sort_values(by='rocp_cl')
ticker_groups[(ticker_groups['Short_%_of_Float'] > 10) & (ticker_groups['Days_to_Cover'] > 10) & (ticker_groups['rocp_cl'] > .05)].sort_values(by='short_close_corr_rocr_20d')


# detect short squeezes based on correlation being around for a few days, in combo with high days to cover
test_corr = all_sh_stocks_df[(all_sh_stocks_df['short_close_corr_10d_EMA'] < -0.7) & (all_sh_stocks_df['Days_to_Cover'] > 10)]

# find historical short squeezes
# correlation detection active, plus shorts going down, plus price going up.
ss = all_sh_stocks_df[(all_sh_stocks_df['Short_%_negative_corr_detection'] == 1) & (all_sh_stocks_df['short_%_rocr_20d'] <= 100)]# & (all_sh_stocks_df['rocp_cl'] >= 0)]

# plot some contiguous stretches
tickers = ss['ticker'].unique().tolist()

random_ticker = np.random.choice(tickers)
print(random_ticker)
df = ss[ss['ticker'] == random_ticker]
df[['short_close_corr_10d_EMA', 'Short_%_negative_corr_detection', 'Adj_Close', 'Short_%_of_Float_10d_EMA', 'Days_to_Cover', 'short_%_rocr_20d', 'rocp_cl']].plot(subplots=True)
plt.show()


for s in all_sh_stocks_df['ticker'].unique():
    df = all_sh_stocks_df[all_sh_stocks_df['ticker'] == s]
    # only get negative correlation with positive price change
    ss = df[(df['Short_%_negative_corr_detection'] == 1) & (df['short_%_rocr_20d'] <= 0) & (df['rocp_cl'] >= 0)]




# train neural net to predict price 20 days out based on this stuff
from keras.layers import Input, Dense, BatchNormalization, Dropout
from keras.models import Model
from keras.optimizers import SGD
import keras.backend as K

train_frac = 0.9
# ignore rows with -np.inf -- cuts data into about 1/3 (2M rows)
# only include small market cap and up, and high short %
nona = all_sh_stocks_df[(all_sh_stocks_df['short_close_corr_10d_EMA'] != -np.inf) & (all_sh_stocks_df['market_cap'] > 300e6) & (all_sh_stocks_df['Short_%_of_Float'] > 10.) & (all_sh_stocks_df['Days_to_Cover'] > 10) & (all_sh_stocks_df['rocp_cl'] > .05)]# & (all_sh_stocks_df['short_%_rocr_20d'].abs() < 1000) & (all_sh_stocks_df['short_close_corr_rocr_20d'].abs() < 1000)]

dates = sorted(nona.index.unique())

train_size = int(train_frac * len(dates))

train_samples = nona.loc[dates[:train_size]]
test_samples = nona.loc[dates[train_size:]]
target_column = '20d_price_change'
# throwout outliers in train
train_samples = train_samples[(train_samples[target_column] > -1) & (train_samples[target_column] < 1)]
target_columns = set([str(i) + 'd_price_change' for i in [5, 10, 20, 40]])
exclude_columns = ['ticker', 'Open', 'Close', 'High', 'Low', 'Volume', 'Adj_Open', 'Adj_Close', 'Adj_High', 'Adj_Low', 'Adj_Volume', 'Dividend', 'Split', 'typical_price']
exclude_columns = set(exclude_columns) | target_columns

# feature_columns = [c for c in all_sh_stocks_df.columns.tolist() if c not in exclude_columns]
# short list for testing
# feature_columns = ['short_%_rocr_20d',
#                     'short_close_corr_rocr_20d',
#                     'short_close_corr_10d_EMA',
#                     'Adj_Close_10d_EMA',
#                     'Short_%_of_Float_10d_EMA',
#                     'Performance_(52-wk)',
#                     'Short_%_of_Float',
#                     'Days_to_Cover',
#                     'rocp_cl',
#                     'rsi_cl',
#                     'macd_cl',
#                     'macdsignal_cl',
#                     'macdhist_cl',
#                     'sar_diff_cl',
#                     'sma_10_cl',
#                     'sma_20_cl',
#                     'sma_30_cl',
#                     'sma_40_cl'
#                     ]


# example of huge short rocr due to < 1% short % of float
# test = train_samples[(train_samples['short_%_rocr_20d'] > 1e6) & (train_samples['ticker'] == 'MTGEP')]
# t = talib.ROCR100(test['Short_%_of_Float_10d_EMA'].values, timeperiod=20)

train_feats = train_samples[feature_columns]
train_targs = train_samples[target_column]
test_feats = test_samples[feature_columns]
test_targs = test_samples[target_column]

from sklearn.preprocessing import RobustScaler, StandardScaler

scaled_train_feats = train_feats.copy()
scaled_test_feats = test_feats.copy()
scalers = []

for c in feature_columns:
    scaler = StandardScaler()# RobustScaler()
    scalers.append(scaler)
    scaled_train_feats[c] = scaler.fit_transform(train_feats[c].values.reshape(-1, 1)).flatten()
    scaled_test_feats[c] = scaler.transform(test_feats[c].values.reshape(-1, 1)).flatten()

scaler = StandardScaler()# RobustScaler()
scalers.append(scaler)
scaled_train_targs = scaler.fit_transform(train_targs.values.reshape(-1, 1)).flatten()
scaled_test_targs = scaler.transform(test_targs.values.reshape(-1, 1)).flatten()

# clip data to boxplot whiskers
# whiskers = {}
# for c in feature_columns:
#     B = plt.boxplot(train_feats[c])
#     min_whisker, max_whisker = [item.get_ydata()[1] for item in B['whiskers']]
#     whiskers[c] = [min_whisker, max_whisker]
#     train_feats[c].clip(lower=min_whisker, upper=max_whisker, inplace=True)
#     test_feats[c].clip(lower=min_whisker, upper=max_whisker, inplace=True)
#
# plt.clf()

# 1677132 down to 1268882
# start = int(train_feats.shape[0] * 0)
# end = int(train_feats.shape[0] * 1)
# tiny_feats = train_feats.iloc[start:end]
# tiny_targs = train_targs.iloc[start:end]

K.clear_session()

inputs = Input(shape=(scaled_train_feats.shape[1],))

# a layer instance is callable on a tensor, and returns a tensor
# x = BatchNormalization()(inputs)
x = Dense(500, activation='elu')(inputs)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
# x = Dense(300, activation='relu')(x)
# x = BatchNormalization()(x)
x = Dense(100, activation='elu')(x)
x = BatchNormalization()(x)
x = Dense(100, activation='elu')(x)
x = BatchNormalization()(x)
# x = Dense(50, activation='relu')(x)
# x = BatchNormalization()(x)
x = Dense(10, activation='elu')(x)
x = BatchNormalization()(x)
predictions = Dense(1, activation='linear')(x)

# This creates a model that includes
# the Input layer and three Dense layers
model = Model(inputs=inputs, outputs=predictions)
sgd = SGD(lr=0.1, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='mse')
history = model.fit(scaled_train_feats, scaled_train_targs, batch_size=20000, epochs=1000, validation_split=0.1)

train_preds = model.predict(scaled_train_feats, batch_size=20000, verbose=1)
test_preds = model.predict(scaled_test_feats, batch_size=20000, verbose=1)

# model.evaluate(scaled_train_feats, scaled_train_targs, batch_size=10000)
# model.evaluate(scaled_test_feats, scaled_test_targs, batch_size=10000)



sample = np.random.choice(np.arange(train_preds.shape[0]), 100000)
plt.scatter(scaled_train_targs[sample], train_preds[sample])
# plt.xlim([-1, 1])
# plt.ylim([-1, 1])
plt.show()

sample = np.random.choice(np.arange(test_preds.shape[0]), 100000)
plt.scatter(test_targs[sample], test_preds[sample])
# plt.xlim([-1, 1])
# plt.ylim([-1, 1])
plt.show()

sample = shorts_nona[shorts_nona['ticker'] == ticker]
plt.scatter(sample['short_%_rocr_20d'], sample['20d_price_change'])
plt.xlim(0, 200)
plt.ylim(-2, 2)
plt.show()

# only "small" market caps 300m to 2b
# https://www.investopedia.com/terms/s/small-cap.asp
short_nona_sm = shorts_nona[(shorts_nona['market_cap'] > 300e6) & (shorts_nona['market_cap'] < 2e9)]
short_nona_sm[['Short_Squeeze_Ranking',
                'Total_Short_Interest',
                'Days_to_Cover',
                'Short_%_of_Float',
                'Performance_(52-wk)',
                '%_Insider_Ownership',
                '%_Institutional_Ownership',
                '%_from_52-wk_High',
                '%_from_200_day_MA',
                '%_from_50_day_MA',
                'Shares_Float',
                'Avg._Daily_Vol.',
                'Shares_Outstanding',
                '%_Change_Mo/Mo',
                '20d_price_change'
                ]].corr()
short_nona_sm[['short_%_rocr_20d', '20d_price_change']].corr()
sample = short_nona_sm.sample(100000)
# ticker = 'AAPL'
# sample = shorts_nona[shorts_nona['ticker'] == ticker]
plt.scatter(sample['short_%_rocr_20d'], sample['20d_price_change'])
plt.xlim(0, 200)
plt.ylim(-2, 2)
plt.show()


# look for correlations between price changes and short squeeze metrics
price_changes = [str(i) + 'd_price_change' for i in [5, 10, 20, 40]]
ss_cols = ['Total_Short_Interest',
            'Days_to_Cover',
            'Short_%_of_Float',
            '%_Insider_Ownership',
            '%_Institutional_Ownership',
            '%_Change_Mo/Mo',
            'score',
            'Short_Squeeze_Ranking']
all_sh_stocks_df[ss_cols + price_changes].corr()

# need to sample down...
df_sample = all_sh_stocks_df.sample(10000)
plt.scatter(df_sample['Short_Squeeze_Ranking'], df_sample['20d_price_change'], alpha=0.01)
plt.show()

# calculate moving average of short interest over 10d SMA
sh_int['CRC']['Short_%_of_Float_10d_SMA'] = talib.SMA(sh_int['CRC']['Short_%_of_Float'].values, timeperiod=10)
sh_int['CRC']['Adj_Close_10d_SMA'] = talib.SMA(sh_int['CRC']['Adj_Close'].values, timeperiod=10)
sh_int['CRC'][['Short_%_of_Float', 'Short_%_of_Float_10d_SMA']].plot()

# look for sections where short % mva and price are (ideally, inversely) correlated
sh_int['CRC'].plot.scatter(x='Short_%_of_Float_10d_SMA', y='Adj_Close')
plt.show()

# essentially, we want to take an arbitrary number of points, calculate correlation, and find where the correlations are largest
# take 10 points at a time and get correlations first, then take parts that have largest correlations, and keep expanding by 5 points at a time until correlation decreases
corr = sh_int['CRC'][['Short_%_of_Float_10d_SMA', 'Adj_Close_10d_SMA']].rolling(window=20).corr()
# rows, columns
f, ax = plt.subplots(nrows=2, ncols=1, sharex=True)
corr.unstack(1)['Short_%_of_Float_10d_SMA']['Adj_Close_10d_SMA'].plot(ax=ax[0])
sh_int['CRC'][['Adj_Close_10d_SMA', 'Short_%_of_Float_10d_SMA']].plot(ax=ax[1])
plt.show()

# in general, when correlation is largely negative for a long time, it's a short or long squeeze

# also look at EMA
sh_int['CRC']['Short_%_of_Float_10d_EMA'] = talib.EMA(sh_int['CRC']['Short_%_of_Float'].values, timeperiod=10)
sh_int['CRC']['Adj_Close_10d_EMA'] = talib.EMA(sh_int['CRC']['Adj_Close'].values, timeperiod=10)
sh_int['CRC'][['Short_%_of_Float', 'Short_%_of_Float_10d_EMA']].plot()

# look for sections where short % mva and price are (ideally, inversely) correlated
sh_int['CRC'].plot.scatter(x='Short_%_of_Float_10d_EMA', y='Adj_Close')
plt.show()

# essentially, we want to take an arbitrary number of points, calculate correlation, and find where the correlations are largest
# take 10 points at a time and get correlations first, then take parts that have largest correlations, and keep expanding by 5 points at a time until correlation decreases
corr = sh_int['CRC'][['Short_%_of_Float_10d_EMA', 'Adj_Close_10d_EMA']].rolling(window=20).corr()
sh_int['CRC']['short_close_corr_10d_EMA'] = corr.unstack(1)['Short_%_of_Float_10d_EMA']['Adj_Close_10d_EMA']

# rows, columns
f, ax = plt.subplots(nrows=2, ncols=1, sharex=True)
sh_int['CRC']['short_close_corr_10d_EMA'].plot(ax=ax[0])
sh_int['CRC'][['Adj_Close_10d_EMA', 'Short_%_of_Float_10d_EMA']].plot(ax=ax[1])
plt.show()

# auto-detect long stretches of negative and positive correlation
thresh = 0.8
rolling = sh_int['CRC']['short_close_corr_10d_EMA'].rolling(window=20).min()
sh_int['CRC']['Short_%_positive_corr_detection'] = rolling > thresh
sh_int['CRC']['Short_%_positive_corr_detection'] = sh_int['CRC']['Short_%_positive_corr_detection'].astype('int16')
sh_int['CRC']['Short_%_positive_corr_detection'].plot()
plt.show()

sh_int['CRC']['Short_%_negative_corr_detection'] = rolling < -thresh
sh_int['CRC']['Short_%_negative_corr_detection'] = sh_int['CRC']['Short_%_negative_corr_detection'].astype('int16')
sh_int['CRC'][['short_close_corr_10d_EMA', 'Short_%_negative_corr_detection', 'Adj_Close', 'Short_%_of_Float_10d_EMA']].plot(subplots=True)
plt.show()


ticker = 'TDOC'
train_frac = 0.8
train_size = int(train_frac * sh_int[ticker].shape[0])
target_column = '20d_price_change'
target_columns = set([str(i) + 'd_price_change' for i in [5, 10, 20, 40]])

for i in [5, 10, 20, 40]:
    sh_int[ticker][str(i) + 'd_price_change'] = sh_int[ticker]['Adj_Close'].pct_change(i).shift(-i)

feature_columns = [c for c in sh_int[ticker] if c not in target_columns]
nona = sh_int[ticker].dropna()
features = nona[feature_columns]
targets = nona[target_column]
train_feats = features[:train_size]
train_targs = targets[:train_size]
test_feats = features[train_size:]
test_targs = targets[train_size:]

rfr = RandomForestRegressor(n_estimators=200, oob_score=True, n_jobs=-1, random_state=42)
rfr.fit(train_feats, train_targs)
rfr.oob_score_


grid = ParameterGrid({'max_features': [6, 13, 20, 50, 100, 150],
                        'max_depth': [5, 10, 20, 50],
                        'min_samples_split': [2, 4, 10],
                        'oob_score': [True],
                        'random_state': [42],
                        'n_jobs': [-1]})

train_scores = []
test_scores = []
oob_scores = []
for g in grid:
    rfr.set_params(**g)
    rfr.fit(train_feats, train_targs)
    train_scores.append(rfr.score(train_feats, train_targs))
    test_scores.append(rfr.score(test_feats, test_targs))
    oob_scores.append(rfr.oob_score_)

# train scores were very high, but test scores horrible

feature_columns = ['short_close_corr_10d_EMA',
                    'Short_%_of_Float_10d_EMA',
                    'Adj_Close_10d_EMA',
                    'Days_to_Cover',
                    'rocr_cl_100',
                    'Adj_Volume',
                    'ultosc',
                    'adosc',
                    'natr']
# need rates of change of corr, float shares, and adj close
sh_int[ticker]['short_close_corr_rocr_20d'] = talib.ROCR100(sh_int[ticker]['short_close_corr_10d_EMA'].values, timeperiod=20)
sh_int[ticker]['short_%_rocr_20d'] = talib.ROCR100(sh_int[ticker]['Short_%_of_Float_10d_EMA'].values, timeperiod=20)
nona = sh_int[ticker].dropna()

features = nona[feature_columns]
targets = nona[target_column]
train_feats = features[:train_size]
train_targs = targets[:train_size]
test_feats = features[train_size:]
test_targs = targets[train_size:]

# rfr = RandomForestRegressor(n_estimators=200, max_depth=5, min_samples_split=2, oob_score=True, n_jobs=-1, random_state=42)
# rfr.fit(train_feats, train_targs)
# rfr.score(test_feats, test_targs)

# when 10-day short % rocr is exceptionally low, returns are high over 20 days
# when rocr is exceptionally high, returns are smaller, but still positive
plt.scatter(nona['short_%_rocr_20d'], targets)
plt.show()


non_neg1_ss_rank = all_sh_stocks_df[all_sh_stocks_df['Short_Squeeze_Ranking'] != -1]
non_neg1_ss_rank[['Short_Squeeze_Ranking'] + price_changes].corr()



#
# # any correlation between mom_cl and 10-day change? -- actually seems to be negative, especially when short % is higher
# all_sh_stocks_df[['mom_cl', '10d_price_change']].corr()
# all_sh_stocks_df[(all_sh_stocks_df['mom_cl'] > 10) & (all_sh_stocks_df['Short_%_of_Float'] > 10)][['mom_cl', '10d_price_change']].corr()
# all_sh_stocks_df[(all_sh_stocks_df['mom_cl'] > 10) & (all_sh_stocks_df['Short_Squeeze_Ranking'] > 1)][['mom_cl', '10d_price_change']].corr()
# all_sh_stocks_df[(all_sh_stocks_df['mom_cl'] > 10)][['mom_cl', '10d_price_change']].corr()
#
# all_sh_stocks_df[['Short_Squeeze_Ranking', '10d_price_change']].corr()
# all_sh_stocks_df[(all_sh_stocks_df['Short_Squeeze_Ranking'] > 2)][['Short_Squeeze_Ranking', '10d_price_change']].corr()
#
#
# all_sh_stocks_df[['Short_Squeeze_Ranking'] + price_changes].corr()
# all_sh_stocks_df[(all_sh_stocks_df['Short_Squeeze_Ranking'] > 2)][['Short_Squeeze_Ranking'] + price_changes].corr()
# all_sh_stocks_df[(all_sh_stocks_df['Short_Squeeze_Ranking'] > 5)][['Short_Squeeze_Ranking'] + price_changes].corr()
# all_sh_stocks_df[(all_sh_stocks_df['mom_cl'] > 10)][['mom_cl'] + price_changes].corr()
#
# # weak correlation with long-term price changes
# all_sh_stocks_df[(all_sh_stocks_df['Short_%_of_Float'] > 0)][['Short_%_of_Float'] + price_changes].corr()
# all_sh_stocks_df[['Days_to_Cover'] + price_changes].corr()
#
# all_sh_stocks_df[['%_from_52-wk_High'] + price_changes].corr()
#
# all_sh_stocks_df[(all_sh_stocks_df['mom_cl'] > 10) & (all_sh_stocks_df['Short_%_of_Float'] > 10)].plot.scatter(x='mom_cl', y='10d_price_change')
#
# # find stocks with top momentum (MOM), also those with top momentum and high amounts of shorting
# top_mom = latest_stocks_df.sort_values(by='mom_cl', ascending=False)
# top_mom[['mom_cl', 'Short_%_of_Float', 'Days_to_Cover']].head(10)
# top_mom[top_mom['Short_%_of_Float'] > 10][['mom_cl', 'Short_%_of_Float', 'Days_to_Cover']].head(10)
#
# top_short = latest_stocks_df.sort_values(by=['Short_%_of_Float'], ascending=False)
# top_short[top_short['mom_cl'] > 3][['mom_cl', 'Short_%_of_Float', 'Days_to_Cover']].head(10)
#
# # find stocks with top shortsqueeze ranking
# top_ssr = latest_stocks_df.sort_values(by='Short_Squeeze_Ranking', ascending=False)
# top_ssr[top_ssr['mom_cl'] > 3][['mom_cl', 'Short_%_of_Float', 'Days_to_Cover', 'Short_Squeeze_Ranking']].head(10)
