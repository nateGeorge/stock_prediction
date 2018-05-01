import pandas as pd

import data_processing as dp

dfs, sh_int, fin_sh = dp.load_stocks(stocks=None)

latest_stocks = []
all_sh_stocks = []
latest_date = sh_int['SPY'].index[-1]
for s in sh_int.keys():
    if sh_int[s].shape[0] == 0:
        print(s, 'is empty')
        continue

    if latest_date != sh_int[s].index[-1]:
        print(s, 'is old')
        continue

    df = sh_int[s].copy()
    df['5d_price_change'] = df['Adj_Close'].pct_change(5).shift(-5)
    df['10d_price_change'] = df['Adj_Close'].pct_change(10).shift(-10)
    df['20d_price_change'] = df['Adj_Close'].pct_change(20).shift(-20)
    df['40d_price_change'] = df['Adj_Close'].pct_change(40).shift(-40)
    df['ticker'] = s
    latest_stocks.append(df.iloc[-1])
    all_sh_stocks.append(df.dropna())

latest_stocks_df = pd.concat(latest_stocks, axis=1).T
latest_stocks_df.set_index('ticker', inplace=True)

all_sh_stocks_df = pd.concat(all_sh_stocks)

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
import matplotlib.pyplot as plt
df_sample = all_sh_stocks_df.sample(10000)
plt.scatter(df_sample['Short_Squeeze_Ranking'], df_sample['20d_price_change'], alpha=0.01)
plt.show()

# calculate moving average of short interest over 10d SMA
import talib
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
