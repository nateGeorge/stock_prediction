"""
for loading and processing stockdata
"""
# core
import sys
import gc
import os
import math

# installed
import numpy as np
import pandas as pd
import deepdish as dd
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
import h5py
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler as SS

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


def load_stocks(stocks=None,
                TAs=True,
                finra_shorts=True,
                short_interest=True,
                verbose=False,
                debug=False,
                earliest_date='20150101',
                TAfunc='create_tas',
                calc_scores=True):
    """
    :param stocks: list of strings; tickers (must be caps), if None, will use all stocks possible
    :param TAs: boolean, if true, calculates technical indicators
    :param shorts: boolean, if true, adds all short data
    :param verbose: boolean, prints more debug if true
    :param earliest_date: if using an abbreviated EOD .h5 file (for quicker
                            loading), provide earliest date
    :param TAfunc: string, function name for TA creation in calculate_ta_signals.py
    :param calc_scores: boolean, if true will calculate custom scoring metric

    :returns: dict of pandas dataframes with tickers as keys,
                dict of dataframes merged with short interest data (sh_int),
                dict of dataframes merged with finra data (fin_sh)
    """
    print('loading stocks...')
    all_stocks_dfs = dlq.load_stocks(verbose=verbose, earliest_date=earliest_date)
    dfs = {}
    existing_stocks = set(all_stocks_dfs.keys())
    if stocks is None:
        stocks = existing_stocks

    for s in stocks:
        if s in existing_stocks:
            dfs[s] = all_stocks_dfs[s]
        else:
            if verbose:
                print('stock', s, 'not in quandl data!')

    ret_stocks = sorted(dfs.keys())  # sometimes some stocks are not in there

    jobs = []
    if TAs:
        print('calculating TAs...')
        with ProcessPoolExecutor(max_workers=None) as executor:
            for s in ret_stocks:
                r = executor.submit(getattr(cts, TAfunc),
                                    dfs[s],
                                    return_df=True)
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

        ss_sh = sse.get_short_interest_data(all_cols=True)
        ss_sh.rename(columns={'Symbol': 'Ticker'}, inplace=True)
        ss_sh_grp = ss_sh.groupby('Ticker')
        sh_stocks = set(ss_sh['Ticker'].unique())
        if debug:
            for s in ret_stocks:
                if s in sh_stocks:
                    sh_int[s] = make_sh_df(s, dfs[s], ss_sh_grp.get_group(s))
        else:
            jobs = []
            with ProcessPoolExecutor() as executor:
                for s in ret_stocks:
                    if s in sh_stocks:
                        r = executor.submit(make_sh_df,
                                            s,
                                            dfs[s],
                                            ss_sh_grp.get_group(s),
                                            verbose,
                                            calc_scores)
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


def make_sh_df(s, df, ss_sh_df, verbose=False, calc_scores=True):
    if verbose:
        print(s)

    new = df[df['Date'] >= ss_sh_df['Date'].min()]
    new = new.merge(ss_sh_df, how='left', on=['Date'])
    # two ticker columns after the merge, we don't need them
    new.drop(['Ticker_x', 'Ticker_y'], axis=1, inplace=True)
    new.ffill(inplace=True)
    new.fillna(-1, inplace=True)
    new.set_index('Date', inplace=True)
    if calc_scores:
        try:
            new['score'] = new.apply(calc_score, axis=1)  # custom scoring metric
            new['score_no_penalty'] = new.apply(calc_score, penalty=False, axis=1)
        except ValueError:  # if all the values are nan, then gives error (e.g. ACFN)
            new['score'] = np.nan
            new['score_no_penalty'] = np.nan

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


def make_all_sh_future(sh_int, future=10, hist_points=40, verbose=False):
    for s in sorted(sh_int.keys()):
        if '10_day_price_diff' not in sh_int[s].columns and sh_int[s].shape[0] > hist_points:
            if verbose:
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
        hist_dates = dates[hist_points:stop] # dates for last of historical points
        dates = dates[hist_points + future:]  # dates for the targets
        return new_feats, new_targs, dates, hist_dates


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


def look_at_ss_rank_plot():
    # make giant combination of all stocks and look at correlation between shortsqueeze rating 10 days and price change 10 days later
    # no correlation, but this isn't super helpful, because it should be the
    # ranking on the day new data comes out vs the price change 10 days later.
    sh_int['A'].columns.tolist().index('Short Squeeze Ranking')  # 145
    all_feats, all_targs = [], []
    for s in sh_int_stocks:
        if sh_int[s].shape[0] > hist_points:  # make_all_sh_future did this too
            # print(s)
            new_feats, new_targs, _, _ = create_hist_feats(sh_int[s]['Short Squeeze Ranking'].values.reshape(-1, 1),
                                                            sh_int[s][targ_col].values,
                                                            sh_int[s].index.values,
                                                            hist_points=1,
                                                            future=10)
            all_feats.append(new_feats)
            all_targs.append(new_targs)

    all_feats_np = np.concatenate(all_feats).flatten()
    all_targs_np = np.concatenate(all_targs).flatten()

    # no correlation, but this isn't super helpful, because it should be the
    # ranking on the day new data comes out vs the price change 10 days later.
    trace = Scattergl(
    x = all_feats_np,
    y = all_targs_np,
    mode = 'markers',
    marker = dict(
        color = '#FFBAD2',
        line = dict(width = 1)
        )
    )
    data = [trace]
    plot(data, filename='webgl')


def check_targets():
    # make sure targets were made correctly
    # still not 100% double checked that targets are correct
    # but future % change is calculated correct
    from plotly import tools
    from plotly.graph_objs import Scatter, Figure, Scattergl
    from plotly.offline import plot
    trace1 = Scatter(
    x=sh_int['A'].index,
    y=sh_int['A']['typical_price']
    )
    trace2 = Scatter(
        x=sh_int['A'].index,
        y=sh_int['A']['10_day_price_diff_pct'],
    )

    targ_col = str(future) + '_day_price_diff_pct'
    feat_cols = sorted(set(sh_int[sh_int_stocks[0]].columns).difference(set([str(future) + '_day_price_diff', targ_col])))
    a_feats, a_targs, a_dates, a_h_dates = create_hist_feats(sh_int['A'][feat_cols].values,
                                                    sh_int['A'][targ_col].values,
                                                    sh_int['A'].index.values,
                                                    hist_points=hist_points,
                                                    future=future)
    trace3 = Scatter(
        x=a_dates,
        y=a_targs,
    )

    trace1b = Scatter(
        x=a_h_dates,
        y=a_feats[:, -1, 12],  # typical price column
    )

    fig = tools.make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.001)

    fig.append_trace(trace1, 1, 1)
    fig.append_trace(trace1b, 2, 1)
    # fig.append_trace(trace2, 2, 1)
    fig.append_trace(trace3, 3, 1)

    fig['layout'].update(height=1500, width=1500)
    plot(fig, filename='simple-subplot')


def make_nn_data(sh_int, hist_points=40, future=10, test_frac=0.15, make_fresh=False, verbose=False):
    # make historical feats for all
    # need to do this in chunks and save it
    # break into 2 chunks
    # uses about 16GB of memory per chunk

    # only uses stocks with short data
    sh_int_stocks = sorted(sh_int.keys())
    targ_col = str(future) + '_day_price_diff_pct'
    feat_cols = sorted(set(sh_int[sh_int_stocks[0]].columns).difference(set([str(future) + '_day_price_diff', targ_col])))
    chunks = 10
    breakpoint = len(sh_int_stocks) // chunks
    fname = big_data_home_dir + 'hist=' + str(hist_points) + 'fut=' + str(future) + 'testfrac=' + str(test_frac) + '/'
    if not os.path.exists(fname):
        os.mkdir(fname)

    for i in range(chunks):
        filename = fname + 'ch_' + str(i) + '.h5'
        if make_fresh and os.path.exists(filename):
            os.remove(filename)

        start = i * breakpoint
        end = (i + 1) * breakpoint
        if i == chunks - 1:  # last chunk
            end = -1

        ch = sh_int_stocks[start:end]
        all_tr_feats, all_tr_targs, all_te_feats, all_te_targs = [], [], [], []
        tr_indices, te_indices = [], []
        # indices where stocks start/end in big dataset, in case want to look at
        # individual stocks later
        st_tr, st_te, end_tr, end_te = 0, 0, 0, 0
        for s in ch:
            if verbose:
                print(s)

            # first make sure we have enough data to make the feats/targs
            if sh_int[s].shape[0] > math.ceil((hist_points + future) / test_frac) + 5:
                new_feats, new_targs, _, _ = create_hist_feats(sh_int[s][feat_cols].values,
                                                                sh_int[s][targ_col].values,
                                                                sh_int[s].index.values,
                                                                hist_points=hist_points,
                                                                future=future)
                tr_idx = int(round((1 - test_frac) * new_feats.shape[0]))
                all_tr_feats.append(new_feats[:tr_idx, :, :])
                all_tr_targs.append(new_targs[:tr_idx])
                all_te_feats.append(new_feats[tr_idx:, :, :])
                all_te_targs.append(new_targs[tr_idx:])
                # keeps track of the indices for
                end_tr = tr_idx
                end_te = new_feats.shape[0] - tr_idx
                tr_indices.append([st_tr, end_tr])
                te_indices.append([st_te, end_te])
                st_tr += end_tr
                st_te += end_te
                # all_dates.append(dates)

        tr_feats = np.concatenate(all_tr_feats)
        tr_targs = np.concatenate(all_tr_targs)
        te_feats = np.concatenate(all_te_feats)
        te_targs = np.concatenate(all_te_targs)
        tr_sizes = np.array(tr_indices)
        te_sizes = np.array(te_indices)

        # z-scaling
        scale_historical_feats(tr_feats)
        scale_historical_feats(te_feats)

        if make_fresh and os.path.exists(filename):
            os.remove(filename)

        f = h5py.File(filename)
        f.create_dataset('tr_feats', data=tr_feats, compression='lzf')
        f.create_dataset('tr_targs', data=tr_targs, compression='lzf')
        f.create_dataset('te_feats', data=te_feats, compression='lzf')
        f.create_dataset('te_targs', data=te_targs, compression='lzf')
        f.create_dataset('tr_indices', data=tr_indices, compression='lzf')
        f.create_dataset('te_indices', data=te_indices, compression='lzf')
        f.close()

        # make list of tickers
        df = pd.DataFrame({'Tickers': ch})
        df.to_hdf(fname + 'ch_' + str(i) + 'stocks.h5',
                    key='data',
                    complib='blosc',
                    complevel=9)
        # save memory
        del df
        gc.collect()
        # didn't work...error about converting to int and too large
        # dd.io.save('ch_{}.h5'.format(i + 1), {'feats': all_feats, 'targs': all_targs, 'stocks': ch}, compression=('blosc', 9))


def scale_historical_feats(feats, multiproc=False):
    # TODO: multithread so it runs faster
    if multiproc:
        cores = os.cpu_count()

        chunksize = feats.shape[0] // (cores)
        chunks = np.split(feats[:chunksize * (cores - 1), :, :], cores - 1)
        print(len(chunks))
        print(feats[np.newaxis, chunksize * (cores - 1):, :, :].shape)
        print(chunks[0].shape)
        # chunks = np.concatenate([chunks, feats[np.newaxis, chunksize * (cores - 1):, :, :]])

        pool = Pool()
        pool.map(scale_it, chunks + [feats[np.newaxis, chunksize * (cores - 1):, :, :]])
        pool.close()
        pool.join()
    else:
        scale_it(feats)


def scale_it(dat, tq=True):
    sh0, sh2 = dat.shape[0], dat.shape[2]
    s = SS(copy=False)  # copy=False does the scaling inplace, so we don't have to make a new list
    if tq:
        it = tqdm(range(sh0))
    else:
        it = range(sh0)
    for j in it:  # timesteps
        for i in range(sh2):  # number of indicators/etc
            _ = s.fit_transform(dat[j, :, i].reshape(-1, 1))[:, 0]


def load_nn_data_one_set(i=0, hist_points=40, future=10, test_frac=0.15):
    # loads just one set of neural net training data out of 10. 'i' specifies which set
    # first get folder name
    fname = big_data_home_dir + 'hist=' + str(hist_points) + 'fut=' + str(future) + 'testfrac=' + str(test_frac) + '/'
    f = h5py.File(fname + 'ch_' + str(i) + '.h5')
    tr_feats = f['tr_feats'][:]
    tr_targs = f['tr_targs'][:]
    te_feats = f['te_feats'][:]
    te_targs = f['te_targs'][:]
    tr_indices = f['tr_indices'][:]
    te_indices = f['te_indices'][:]
    f.close()
    stocks = pd.read_hdf(fname + 'ch_' + str(i) + 'stocks.h5')
    stocks = stocks['Tickers'].values
    return tr_feats, tr_targs, te_feats, te_targs, tr_indices, te_indices, stocks


def calc_score(df, penalty=True):
    """
    calculates custom scoring metric based on short squeeze ranking, etc
    """
    # might want to also check for -1 or whatever missing values are filled in with
    score = 0
    if pd.notna(df['Short_Squeeze_Ranking']):
        score = df['Short_Squeeze_Ranking'] * 0.4
    if pd.notna(df['Short_%_of_Float']):
        score += df['Short_%_of_Float'] * 0.1
    if pd.notna(df['Days_to_Cover']):
        score += df['Days_to_Cover'] * 0.2
    if pd.notna(df['%_from_52-wk_High']):
        score += df['%_from_52-wk_High'] * 0.1

    score += df['macd_tp'] * 0.2
    if penalty and df['macd_tp'] <= 0.05:
        score -= 50

    return score


if __name__ == "__main__":
    pass
    # dfs, sh_int, fin_sh = load_stocks(stocks=None, finra_shorts=False, short_interest=False, verbose=True)
    # short_stocks = sse.get_stocks()
    # dfs, sh_int, fin_sh = load_stocks(stocks=short_stocks, verbose=True)
    # future = 10
    # hist_points = 40
    # make_all_sh_future(sh_int, future=future, hist_points=hist_points, verbose=False)
    # # del dfs
    # # del fin_sh
    # gc.collect()
    #
    # make_nn_data(sh_int, hist_points=hist_points, future=future, make_fresh=True)
    #
    # tr_feats, tr_targs, te_feats, te_targs, tr_indices, te_indices, stocks = load_nn_data_one_set(i=0, hist_points=hist_points, future=future)

    # needed this one time to make plots...not sure if needed again
    # for s in sh_int_stocks:
    #     sh_int[s]['Date'] = pd.to_datetime(sh_int[s]['Date'])
    #     sh_int[s].set_index('Date', inplace=True)


    # test out neural net






    # EDA('CVGW', dfs, sh_int, fin_sh)
