"""
This is intended to be plug-and-play short squeeze analysis.  Generates top picks and plots.

Need to add in earnings dates into the mix.  Stocks with earnings coming up should be looked into more.
"""

# installed
import pandas as pd
import numpy as np
import talib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import ParameterGrid
import matplotlib.pyplot as plt

# custom
import data_processing as dp


def load_stocks_calculate_short_corr(only_latest=False):
    """
    only_latest: boolean; if True, will keep only the latest 50 days to speed up procesing
    """
    dfs, sh_int, fin_sh = dp.load_stocks(stocks=None)
    if only_latest:
        earliest = sh_ind.index[-50]
        sh_int = sh_int[sh_int.index > earliest]

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
