"""
for loading and processing stockdata
"""
# core
import sys
import gc

# installed
import numpy as np
import pandas as pd
import deepdish as dd
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor

# custom
import dl_quandl_EOD as dlq
import calculate_ta_signals as cts
sys.path.append('/home/nate/github/scrape_stocks/')
import scrape_finra_shorts as sfs
import short_squeeze_eda as sse

# yeah yeah, should be caps
indicators = ['bband_u_cl', # bollinger bands
             'bband_m_cl',
             'bband_l_cl',
             'bband_u_tp',
             'bband_m_tp',
             'bband_l_tp',
             'bband_u_cl_diff',
             'bband_m_cl_diff',
             'bband_l_cl_diff',
             'bband_u_cl_diff_hi',
             'bband_l_cl_diff_lo',
             'bband_u_tp_diff',
             'bband_m_tp_diff',
             'bband_l_tp_diff',
             'bband_u_tp_diff_hi',
             'bband_l_tp_diff_lo',
             'dema_cl',
             'dema_tp',
             'dema_cl_diff',
             'dema_tp_diff',
             'ema_cl',
             'ema_tp',
             'ema_cl_diff',
             'ema_tp_diff',
             'ht_tl_cl',
             'ht_tl_tp',
             'ht_tl_cl_diff',
             'ht_tl_tp_diff',
             'kama_cl',
             'kama_tp',
             'kama_cl_diff',
             'kama_tp_diff',
            #  'mama_cl',  # having problems with these
            #  'mama_tp',
            #  'fama_cl',
            #  'fama_tp',
            #  'mama_cl_osc',
            #  'mama_tp_osc',
             'midp_cl',
             'midp_tp',
             'midp_cl_diff',
             'midp_tp_diff',
             'midpr',
             'midpr_diff',
             'sar',
             'sar_diff',
             'tema_cl',
             'tema_tp',
             'tema_cl_diff',
             'tema_tp_diff',
             'trima_cl',
             'trima_tp',
             'trima_cl_diff',
             'trima_tp_diff',
             'wma_cl',
             'wma_tp',
             'wma_cl_diff',
             'wma_tp_diff',
             'adx',
             'adxr',
             'apo_cl',
             'apo_tp',
             'arup', # aroon
             'ardn',
             'aroonosc',
             'bop',
             'cci',
             'cmo_cl',
             'cmo_tp',
             'dx',
             'macd_cl',
             'macdsignal_cl',
             'macdhist_cl',
             'macd_tp',
             'macdsignal_tp',
             'macdhist_tp',
             'mfi',
             'mdi',
             'mdm',
             'mom_cl',
             'mom_tp',
             'pldi',
             'pldm',
             'ppo_cl',
             'ppo_tp',
             'roc_cl',
             'roc_tp',
             'rocp_cl',
             'rocp_tp',
             'rocr_cl',
             'rocr_tp',
             'rocr_cl_100',
             'rocr_tp_100',
             'rsi_cl',
             'rsi_tp',
             'slowk', # stochastic oscillator
             'slowd',
             'fastk',
             'fastd',
             'strsi_cl_k',
             'strsi_cl_d',
             'strsi_tp_k',
             'strsi_tp_d',
             'trix_cl',
             'trix_tp',
             'ultosc',
             'willr',
             'ad',
             'adosc',
             'obv_cl',
             'obv_tp',
             'atr',
             'natr',
             'trange',
             'ht_dcp_cl',
             'ht_dcp_tp',
             'ht_dcph_cl',
             'ht_dcph_tp',
             'ht_ph_cl',
             'ht_ph_tp',
             'ht_q_cl',
             'ht_q_tp',
             'ht_s_cl',
             'ht_s_tp',
             'ht_ls_cl',
             'ht_ls_tp',
             'ht_tr_cl',
             'ht_tr_tp'
             ]
# data too big for main drive
big_data_home_dir = '/media/nate/data_lake/stock_data/'


def load_stocks(stocks=['NAVI', 'EXAS'],
                TAs=True,
                finra_shorts=True,
                short_interest=True,
                verbose=False,
                earliest_date='20150101'):
    """
    :param stocks: list of strings; tickers (must be caps)
    :param TAs: boolean, if true, calculates technical indicators
    :param shorts: boolean, if true, adds all short data
    :param verbose: boolean, prints more debug if true
    :param earliest_date: if using an abbreviated EOD .h5 file (for quicker
                            loading), provide earliest date

    :returns: dict of pandas dataframes with tickers as keys,
                dict of dataframes merged with short interest data (sh_int),
                dict of dataframes merged with finra data (fin_sh)
    """
    print('loading stocks...')
    dfs = dlq.load_stocks(stocks=stocks, verbose=verbose, earliest_date=earliest_date)
    ret_stocks = sorted(dfs.keys())  # sometimes some stocks are not in there

    jobs = []
    if TAs:
        print('calculating TAs...')
        with ProcessPoolExecutor(max_workers=None) as executor:
            for s in ret_stocks:
                r = executor.submit(cts.create_tas,
                                    dfs[s],
                                    return_df=True,
                                    verbose=verbose)
                jobs.append((s, r))

        for s, r in jobs:
            res = r.result()
            if res is not None:
                dfs[s] = res
            else:
                print('result is None for', s)

        del jobs
        gc.collect()

    sh_int = {}
    fin_sh = {}
    # not sure if processpool helping here at all...maybe even want to do
    # thread pool, or just loop it
    if finra_shorts:
        for s in ret_stocks:
            dfs[s].reset_index(inplace=True)

        print('getting finra shorts and merging...')
        finra_sh_df = sfs.load_all_data()
        finra_sh_df.rename(columns={'Symbol': 'Ticker'}, inplace=True)
        fn_stocks = set(finra_sh_df['Ticker'].unique())
        fn_grp = finra_sh_df.groupby(['Ticker', 'Date']).sum()
        jobs = []
        with ProcessPoolExecutor() as executor:
            for s in ret_stocks:
                if s in fn_stocks:
                    r = executor.submit(make_fn_df, s, dfs[s], fn_grp.loc[s])
                    jobs.append((s, r))

        for s, r in jobs:
            res = r.result()
            if res is not None:
                fin_sh[s] = res
            else:
                print('result is None for', s)

        del jobs
        gc.collect()

    if short_interest:
        print('getting short interest and merging...')
        if 'Date' not in dfs[ret_stocks[0]].columns:
            for s in ret_stocks:
                dfs[s].reset_index(inplace=True)

        ss_sh = sse.get_short_interest_data()
        ss_sh.rename(columns={'Symbol': 'Ticker'}, inplace=True)
        ss_sh_grp = ss_sh.groupby('Ticker')
        sh_stocks = set(ss_sh['Ticker'].unique())
        jobs = []
        with ProcessPoolExecutor() as executor:
            for s in ret_stocks:
                if s in sh_stocks:
                    r = executor.submit(make_sh_df,
                                        s,
                                        dfs[s],
                                        ss_sh_grp.get_group(s),
                                        verbose)
                    jobs.append((s, r))

        for s, r in jobs:
            res = r.result()
            if res is not None:
                sh_int[s] = res
            else:
                print('result is None for', s)

        del jobs
        gc.collect()

    return dfs, sh_int, fin_sh


def save_dfs(dfs, sh_int, fin_sh):
    dd.io.save('proc_dfs.h5', {'dfs': dfs, 'sh_int': sh_int, 'fin_sh': fin_sh}, compression=('blosc', 9))


def load_dfs():
    dat = dd.io.load('proc_dfs.h5')
    return dat['dfs'], dat['sh_int'], dat['fin_sh']


def make_sh_df(s, df, ss_sh_df, verbose):
    if verbose:
        print(s)
    
    new = df[df['Date'] >= ss_sh_df['Date'].min()]
    new = new.merge(ss_sh_df, how='left', on=['Date'])
    new.ffill(inplace=True)
    new.fillna(-1, inplace=True)
    new.set_index('Date', inplace=True)
    return new


def make_fn_df(s, df, fn_df_s):
    fn_df_s.reset_index(inplace=True)
    fn_df_s['Ticker'] = s
    new_df = df.merge(fn_df_s, on=['Ticker', 'Date'])
    return new_df


def make_gain_targets(df, future=10, return_df=False):
    """
    :param df: pandas dataframe with typical_price
    :param future: number of days in the future to calculate % return,
                    defaults to 10 because the short data is updated twice a month
    """
    col = str(future) + '_day_price_diff'
    df[col] = df['typical_price'].copy()
    df[col] = np.hstack((np.repeat(df[col].iloc[future], future),
                        df['typical_price'].iloc[future:].values - df['typical_price'].iloc[:-future].values))
    df[str(future) + '_day_price_diff_pct'] = df[col] / np.hstack((np.repeat(df['typical_price'].iloc[future], future),
                                                                    df['typical_price'].iloc[:-future].values))
    # drop the first future points because they are all return of 100%
    df = df.iloc[future:]
    if return_df:
        return df


def make_all_sh_future(sh_int, future=10, hist_points=40):
    for s in sorted(sh_int.keys()):
        if '10_day_price_diff' not in sh_int[s].columns and sh_int[s].shape[0] > hist_points:
            print(s)
            make_gain_targets(sh_int[s], future=future)


def create_hist_feats(features, targets, dates, hist_points=40, future=10, make_all=False):
    # make historical features
    new_feats = []
    stop = features.shape[0] - future
    if make_all:
        stop += future

    for i in range(hist_points, stop):
        new_feats.append(features[i - hist_points:i, :])

    new_feats = np.array(new_feats)
    if make_all:
        new_targs = targets[hist_points + future:]
        return new_feats[:-future], targets, new_feats[-future:]
    else:
        new_targs = targets[hist_points + future:]
        dates = dates[hist_points + future:]  # dates for the targets
        return new_feats, new_targs, dates



def EDA(s=None, dfs=None, sh_int=None, fin_sh=None):
    # just encapsulating the first EDA on exas and navi
    return_none = False
    if s is not None:
        if s not in dfs.keys():
            print('not in the main datafile keys')
            return_none = True

        if s not in sh_int.keys():
            print('not in the short interest keys')
            return_none = True

        if s not in fin_sh.keys():
            print('not in the finra short keys')
            return_none = True

        if return_none:
            return

    if dfs is None:
        dfs, sh_int, fin_sh = load_stocks()
        # make_gain_targets(sh_int['NAVI'])
        sh_s = sh_int['EXAS']
        make_gain_targets(sh_s)
        #new = actually sh_int['NAVI']
        fn_s = fin_sh['EXAS']
        make_gain_targets(fn_s)
    else:
        sh_s = sh_int[s]
        fn_s = fin_sh[s]

    if '10_day_price_diff' not in sh_s.columns:
        make_gain_targets(sh_s)
    if '10_day_price_diff' not in fn_s.columns:
        make_gain_targets(fn_s)

    # short interest
    f = plt.figure()
    ax = plt.gca()
    sh_s.plot(y='Short Squeeze Ranking', ax=ax)
    sh_s.plot(y='typical_price', ax=ax, secondary_y=True)
    plt.show()

    f = plt.figure()
    ax = plt.gca()
    sh_s.plot(y='Short % of Float', ax=ax)
    sh_s.plot(y='typical_price', ax=ax, secondary_y=True)
    plt.show()

    f = plt.figure()
    ax = plt.gca()
    sh_s.plot(y='Days to Cover', ax=ax)
    sh_s.plot(y='typical_price', ax=ax, secondary_y=True)
    plt.show()

    sh_s.plot(x='Short Squeeze Ranking', y='5_day_price_diff_pct', kind='scatter')
    plt.show()

    sh_s.plot(x='Short % of Float', y='5_day_price_diff_pct', kind='scatter')
    plt.show()

    sh_s.plot(x='Days to Cover', y='5_day_price_diff_pct', kind='scatter')
    plt.show()

    # FINRA
    f = plt.figure()
    ax = plt.gca()
    fn_s.plot(y='ShortVolume', ax=ax)
    fn_s.plot(y='typical_price', ax=ax, secondary_y=True)
    plt.show()

    fn_s.plot(x='ShortVolume', y='5_day_price_diff_pct', kind='scatter')
    plt.show()


def make_filename(s, future, hist_points, full_train=False):
    if full_train:
        return big_data_home_dir + 'full_train_' + '_'.join([s, 'h=' + str(hist_points), 'f=' + str(future)])

    return big_data_home_dir + 'data/nn_feats_targs/poloniex/' + '_'.join([s, 'h=' + str(hist_points), 'f=' + str(future)])


def prep_nn_data(df,
                make_fresh=False,
                skip_load=False,
                save_scalers=False,
                future=24,
                hist_points=480):
    """
    :param df: pandas dataframe with stock data
    :param make_fresh: creates new files even if they exist
    :param skip_load: if files already exist, just returns all Nones
    :param save_scalers: will save the StandardScalers used.  necessary to do
                        live predictions
    :param future: number of days in the future to predict
    :param hist_points: number of history points to use

    :returns:
    """
    datafile = make_filename(s, future, resamp, hist_points)
    if os.path.exists(datafile) and not make_fresh:
        if skip_load:
            print('files exist, skipping')
            return None, None, None, None

        print('loading...')
        f = h5py.File(datafile, 'r')
        train_feats = f['xform_train'][:]
        test_feats = f['xform_test'][:]
        train_targs = f['train_targs'][:]
        test_targs = f['test_targs'][:]
        dates = f['dates'][:]
        f.close()
    else:
        print('creating new...')
        df = pe.read_trade_hist(mkt)
        # resamples to the hour if H, T is for minutes, S is seconds
        rs_full = dp.resample_ohlc(df, resamp=resamp)
        del df
        gc.collect()
        rs_full = dp.make_mva_features(rs_full)
        bars = cts.create_tas(bars=rs_full, verbose=True)
        del rs_full
        gc.collect()
        # make target columns
        col = str(future) + '_' + resamp + '_price_diff'
        bars[col] = bars['typical_price'].copy()
        bars[col] = np.hstack((np.repeat(bars[col].iloc[future], future), bars['typical_price'].iloc[future:].values - bars['typical_price'].iloc[:-future].values))
        bars[str(future) + '_' + resamp + '_price_diff_pct'] = bars[col] / np.hstack((np.repeat(bars['typical_price'].iloc[future], future), bars['typical_price'].iloc[:-future].values))
        # drop first 'future' points because they are repeated
        # also drop first 1000 points because usually they are bogus
        bars = bars.iloc[future + 1000:]
        dates = list(map(lambda x: x.value, bars.index))  # in microseconds since epoch
        if bars.shape[0] < 1000:
            print('less than 1000 points, skipping...')
            return None, None, None, None

        feat_cols = indicators + ['mva_tp_24_diff', 'direction_volume', 'volume', 'high', 'low', 'close', 'open']
        features = bars[feat_cols].values
        targets = bars[str(future) + '_' + resamp + '_price_diff_pct'].values
        del bars
        gc.collect()

        new_feats, targets, dates = create_hist_feats(features, targets, dates, hist_points=hist_points, future=future)
        test_size=5000
        test_frac=0.2
        scale_historical_feats(new_feats)  # does scaling in-place, no returned values
        # in case dataset is too small for 5k test points, adjust according to test_frac
        if new_feats.shape[0] * test_frac < test_size:
            test_size = int(new_feats.shape[0] * test_frac)

        train_feats = new_feats[:-test_size]
        train_targs = targets[:-test_size]
        test_feats = new_feats[-test_size:]
        test_targs = targets[-test_size:]

        # del targets
        # del new_feats
        # gc.collect()

        f = h5py.File(datafile, 'w')
        f.create_dataset('xform_train', data=train_feats, compression='lzf')
        f.create_dataset('xform_test', data=test_feats, compression='lzf')
        f.create_dataset('train_targs', data=train_targs, compression='lzf')
        f.create_dataset('test_targs', data=test_targs, compression='lzf')
        f.create_dataset('dates', data=dates, compression='lzf')
        f.close()

    return train_feats, test_feats, train_targs, test_targs, dates


if __name__ == "__main__":
    short_stocks = sse.get_stocks()
    dfs, sh_int, fin_sh = load_stocks(stocks=short_stocks, verbose=True)
    sh_int_stocks = sorted(sh_int.keys())
    future = 10
    hist_points = 40
    make_all_sh_future(sh_int, future=future, hist_points=hist_points)
    del dfs
    del fin_sh
    gc.collect()

    # make historical feats for all
    targ_col = str(future) + '_day_price_diff_pct'
    feat_cols = sorted(set(sh_int[sh_int_stocks[0]].columns).difference(set([str(future) + '_day_price_diff', targ_col, 'Ticker'])))
    all_feats, all_targs = [], []
    # for s in sh_int_stocks:
    #     print(s)
    #     if sh_int[s].shape[0] > hist_points:
    #         new_feats, new_targs, _ = create_hist_feats(sh_int[s][feat_cols].values,
    #                                                         sh_int[s][targ_col].values,
    #                                                         sh_int[s].index.values,
    #                                                         hist_points=hist_points,
    #                                                         future=future)
    #         all_feats.append(new_feats)
    #         all_targs.append(new_targs)
            # all_dates.append(dates)

    # make giant combination of all stocks


    # EDA('CVGW', dfs, sh_int, fin_sh)
