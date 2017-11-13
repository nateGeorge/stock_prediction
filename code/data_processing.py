"""
for loading and processing stockdata
"""
# core
import sys

# installed
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# custom
import dl_quandl_EOD as dlq
import calculate_ta_signals as cts
sys.path.append('/home/nate/github/scrape_stocks/')
import scrape_finra_shorts as sfs
import short_squeeze_eda as sse


def load_stocks(stocks=['NAVI', 'EXAS'], TAs=True, finra_shorts=True, short_interest=True):
    """
    :param stocks: list of strings; tickers (must be caps)
    :param TAs: boolean, if true, calculates technical indicators
    :param shorts: boolean, if true, adds all short data

    :returns: dict of pandas dataframes with tickers as keys
    """
    dfs = dlq.load_stocks(stocks=stocks)
    ret_stocks = sorted(dfs.keys())  # sometimes some stocks are not in there
    for s in ret_stocks:
        dfs[s].reset_index(inplace=True, drop=True)

    if TAs:
        for s in ret_stocks:
            cts.create_tas(dfs[s])

    if finra_shorts:
        finra_shorts = sfs.load_all_data()
        finra_shorts.rename(columns={'Symbol': 'Ticker'}, inplace=True)
        fn_grp = finra_shorts.groupby(['Ticker', 'Date']).sum()

    if short_interest:
        ss_sh = sse.get_short_interest_data()
        ss_sh.rename(columns={'Symbol': 'Ticker'}, inplace=True)

    sh_int = {}
    fin_sh = {}
    for s in ret_stocks:
        new = dfs[s][dfs[s]['Date'] >= ss_sh[ss_sh['Ticker'] == s]['Date'].min()]
        new = new.merge(ss_sh, how='left', on=['Ticker', 'Date'])
        new.ffill(inplace=True)
        new.set_index('Date', inplace=True)
        sh_int[s] = new
        fn = fn_grp.loc[s]
        fn.reset_index(inplace=True)
        fn['Ticker'] = s
        new_fn = dfs[s].merge(fn, on=['Ticker', 'Date'])
        fin_sh[s] = new_fn

    return dfs, sh_int, fin_sh


def make_gain_targets(df, future=5):
    """
    :param df: pandas dataframe with typical_price
    :param future: number of days in the future to calculate % return
    """
    col = str(future) + '_day_price_diff'
    df[col] = df['typical_price'].copy()
    df[col] = np.hstack((np.repeat(df[col].iloc[future], future),
                        df['typical_price'].iloc[future:].values - df['typical_price'].iloc[:-future].values))
    df[str(future) + '_day_price_diff_pct'] = df[col] / np.hstack((np.repeat(df['typical_price'].iloc[future], future),
                                                                    df['typical_price'].iloc[:-future].values))
    # drop the first future points because they are all return of 100%
    df = df.iloc[future:]


def EDA(dfs=None, sh_int=None, fin_sh=None):
    # just encapsulating the first EDA on exas and navi
    dfs, sh_int, fin_sh = load_stocks()

    make_gain_targets(sh_int['EXAS'])
    make_gain_targets(sh_int['NAVI'])

    new = sh_int['EXAS']
    #new = sh_int['NAVI']
    f = plt.figure()
    ax = plt.gca()
    new.plot(y='Short Squeeze Ranking', ax=ax)
    new.plot(y='typical_price', ax=ax, secondary_y=True)
    plt.show()

    f = plt.figure()
    ax = plt.gca()
    new.plot(y='Short % of Float', ax=ax)
    new.plot(y='typical_price', ax=ax, secondary_y=True)
    plt.show()

    f = plt.figure()
    ax = plt.gca()
    new.plot(y='Days to Cover', ax=ax)
    new.plot(y='typical_price', ax=ax, secondary_y=True)
    plt.show()


    f = plt.figure()
    ax = plt.gca()
    new.plot(y='Days to Cover', ax=ax)
    new.plot(y='typical_price', ax=ax, secondary_y=True)
    plt.show()


    new.plot(x='Short Squeeze Ranking', y='5_day_price_diff_pct', kind='scatter')
    plt.show()

    new.plot(x='Short % of Float', y='5_day_price_diff_pct', kind='scatter')
    plt.show()

    new.plot(x='Days to Cover', y='5_day_price_diff_pct', kind='scatter')
    plt.show()

    # FINRA
    new = fin_sh['EXAS']
    make_gain_targets(new)
    f = plt.figure()
    ax = plt.gca()
    new.plot(y='ShortVolume', ax=ax)
    new.plot(y='typical_price', ax=ax, secondary_y=True)
    plt.show()

    new.plot(x='ShortVolume', y='5_day_price_diff_pct', kind='scatter')
    plt.show()

short_stocks = sse.get_stocks()
dfs, sh_int, fin_sh = load_stocks(stocks=short_stocks)
