# ported from 3-20-2018_binary_classification_prototype

import pandas as pd
import numpy as np

import sys
sys.path.append('../code')
import data_processing as dp

from concurrent.futures import ProcessPoolExecutor

def get_top_stocks(num=None, indices=True):
    """
    gets top stocks from the volatility screen

    returns: list of string stock tickers
    """
    sorted_old = pd.read_csv('old_good_sorted_by_ewm.csv')
    top_stocks = sorted_old.loc[:, 'ticker'].tolist()
    if num is not None:
        top_stocks = top_stocks[:num]
    if indices:
        top_stocks += ['SPY', 'UPRO', 'QQQ', 'TQQQ', 'DIA', 'UBT']

    return top_stocks


def get_latest_earliest_date(dfs, verbose=False):
    """
    gets the earlies dates from a bunch of dataframes, and returns the
    latest of these dates.

    dfs: dictionary of stock dataframes

    returns: timestamp and dataframe of earliest dates with tickers
    """
    earliest_dates = []
    for s in dfs.keys():
        earliest_date = dfs[s].index.min()
        if verbose:
            print(s, earliest_date)

        earliest_dates.append(earliest_date)

    earliest_dates_df = pd.DataFrame({'earliest_date': earliest_dates,
                                        'ticker': list(dfs.keys())})
    earliest_dates_df.sort_values(by='earliest_date', inplace=True)

    return max(earliest_dates), earliest_dates_df


def drop_cols(dfs, ignore_all=False):
    """
    drops columns that are of no use
    if ignore_all is True, will ignore many columns

    found TAS not bound to range -- from EDA short data 1-2-2018.ipynb file
    """

    # some columns are ignored because they have large outliers
    # that can't really be ignored
    ignore_cols = ['Open',
                    'Close',
                    'Low',
                    'High',
                    'Volume',
                    'Dividend',
                    'Split',
                    'Ticker',
                    'Avg._Daily_Vol.']
    if ignore_all:
        ignore_cols += ['bband_u_cl',
                        'bband_m_cl',
                        'bband_l_cl',
                        'bband_u_tp',
                        'bband_m_tp',
                        'bband_l_tp',
                        'dema_cl',
                        'dema_tp',
                        'ema_cl',
                        'ema_tp',
                        'ht_tl_cl',
                        'ht_tl_tp',
                        'kama_cl',
                        'kama_tp',
                        'mavp_cl',
                        'mavp_tp',
                        'midp_cl',
                        'midp_tp',
                        'midpr',
                        'sar',
                        'sma_10_cl',
                        'sma_10_tp',
                        'sma_20_cl',
                        'sma_20_tp',
                        'sma_30_cl',
                        'sma_30_tp',
                        'sma_40_cl',
                        'sma_40_tp',
                        'tema_cl',
                        'tema_tp',
                        'trima_cl',
                        'trima_tp',
                        'wma_cl',
                        'wma_tp',
                        'mdm',
                        'pldm',
                        'rocr_cl_100', # already have the same data /100 as rocr
                        'rocr_tp_100',
                        'atr',
                        'natr',
                        'trange',
                        'Shares_Float',
                        'Shares_Outstanding']

    ignore_cols = set(ignore_cols)

    first = list(dfs.keys())[0]
    keep_cols = [c for c in dfs[first].columns.tolist() if c not in ignore_cols]
    for s in dfs.keys():
        dfs[s] = dfs[s][keep_cols]


def get_no_info_rate(targets, targ_labels):
    """
    gets no information rate over all targets
    """
    targ_df = pd.DataFrame(targets, columns=targ_labels)
    tot_0 = 0
    tot_1 = 0
    for c in targ_df.columns:
        vc = targ_df[c].value_counts()
        tot_0 += vc[0]
        tot_1 += vc[1]

    if tot_0 > tot_1:
        print('no info rate:', str(tot_0 / (tot_0 + tot_1)))
    else:
        print('no info rate:', str(tot_1 / (tot_0 + tot_1)))


def make_ohlcv_feats_targs_one(s,
                                df,
                                past_cols,
                                fut_cols,
                                past_periods,
                                future_days,
                                threshold=0.05,
                                verbose=False):
    if verbose:
        print(s)

    epsilon = 0.001
    past_pct_change_dict = {}
    fut_pct_change_dict = {}
    for c in past_cols:
        for p in past_periods:
            past_pct_change_dict[c + '_pct_change_p=' + str(p)] = []

    for c in fut_cols:
        for f in future_days:
            fut_pct_change_dict[c + '_pct_change_f=' + str(f) + '_gt_' + str(threshold)] = []
            fut_pct_change_dict[c + '_pct_change_f=' + str(f) + '_lt_-' + str(threshold)] = []

    last_idx = df.shape[0] - 1
    for i, (index, r) in enumerate(df.iterrows()):
        # create time-lagged percent difference features for OHLCV (and maybe eventually TAs) not bound to a range
        for c in past_cols:
            for p in past_periods:
                if i >= p: # if on day 2 (index 1), then the lag would be current - 0
                    old = df[c].iloc[i-p] + epsilon
                    new = df[c].iloc[i]
                    pct_change = (new - old) / old
                    past_pct_change_dict[c + '_pct_change_p=' + str(p)].append(pct_change)
                else:
                    past_pct_change_dict[c + '_pct_change_p=' + str(p)].append(np.nan)
        # create future targets for each of future_days
        for c in fut_cols:
            for f in future_days:
                if i + f <= last_idx: # if on day 2 (index 1), then the lag would be current - 0
                    old = df[c].iloc[i] + epsilon
                    new = df[c].iloc[i + f]
                    pct_change = (new - old) / old
                    if pct_change >= threshold:
                        fut_pct_change_dict[c + '_pct_change_f=' + str(f) + '_gt_' + str(threshold)].append(1)
                    else:
                        fut_pct_change_dict[c + '_pct_change_f=' + str(f) + '_gt_' + str(threshold)].append(0)
                    if pct_change <= -threshold:
                        fut_pct_change_dict[c + '_pct_change_f=' + str(f) + '_lt_-' + str(threshold)].append(1)
                    else:
                        fut_pct_change_dict[c + '_pct_change_f=' + str(f) + '_lt_-' + str(threshold)].append(0)
                else:
                    fut_pct_change_dict[c + '_pct_change_f=' + str(f) + '_gt_' + str(threshold)].append(np.nan)
                    fut_pct_change_dict[c + '_pct_change_f=' + str(f) + '_lt_-' + str(threshold)].append(np.nan)

    # add to dataframe
    df = df.assign(**past_pct_change_dict)
    df = df.assign(**fut_pct_change_dict)
    df.drop(past_cols, inplace=True, axis=1)
    return df


def make_ohlcv_feats_targs_multithread(dfs,
                           past_periods=[1, 2, 3, 5, 10, 20, 30, 50, 100],
                           future_days=range(1, 11),
                           threshold=0.05,
                           verbose=True,
                           debug=False):
    # columns for past data creation
    past_cols = ['Adj_Open', 'Adj_Close', 'Adj_High', 'Adj_Low', 'Adj_Volume', 'typical_price']
    # columns used for future targets
    fut_cols = ['Adj_Close']

    if debug:
        for s in dfs.keys():
            if verbose:
                print(s)

            dfs[s] = make_ohlcv_feats_targs_one(dfs[s], past_cols, fut_cols, past_periods, future_days)

        return

    jobs = []
    with ProcessPoolExecutor(max_workers=None) as executor:
        for s in dfs.keys():
            r = executor.submit(make_ohlcv_feats_targs_one,
                                s,
                                dfs[s],
                                past_cols,
                                fut_cols,
                                past_periods,
                                future_days,
                                threshold,
                                verbose)
            jobs.append((s, r))

    for s, r in jobs:
        res = r.result()
        if res is not None:
            dfs[s] = res
        else:
                print('result is None for', s)

        # no need to return anything, all updates made in-place


def get_features_targets_wide(dfs, train_frac=0.9):
    """
    creates train and test features out of the stock dfs.
    dfs should already have targets made.

    stacks dfs horizontally for a wide dataset (many features and targets, not
    a lot of data points)
    """
    feat_labels = []
    targ_labels = []
    features = None
    for s in dfs.keys():
        targ_cols = [c for c in dfs[s].columns if 'f=' in c]
        feat_cols = [c for c in dfs[s].columns if c not in set(targ_cols)]
        no_missing = dfs[s].dropna()
        s_features = no_missing[feat_cols]
        feat_labels.extend([s + '_' + c for c in s_features.columns.tolist()])
        s_targets = no_missing[targ_cols]
        targ_labels.extend([s + '_' + c for c in s_targets.columns.tolist()])
        if features is None:
            features = s_features.values
            targets = s_targets.values
        else:
            features = np.hstack((features, s_features))
            targets = np.hstack((targets, s_targets))

    train_size = int(features.shape[0] * train_frac)
    train_features = features[:train_size]
    test_features = features[train_size:]
    train_targets = targets[:train_size]
    test_targets = targets[train_size:]

    return train_features, test_features, train_targets, test_targets, feat_labels, targ_labels


def get_features_targets_deep(dfs, train_frac=0.9):
    """
    creates train and test features out of the stock dfs.
    dfs should already have targets made.

    stacks dfs horizontally for a wide dataset (many features and targets, not
    a lot of data points)
    """
    feat_labels = []
    targ_labels = []
    train_features = None

    for s in dfs.keys():
        targ_cols = [c for c in dfs[s].columns if 'f=' in c]
        feat_cols = [c for c in dfs[s].columns if c not in set(targ_cols)]
        no_missing = dfs[s].dropna()
        s_features = no_missing[feat_cols]
        feat_labels = [c for c in s_features.columns.tolist()]
        s_targets = no_missing[targ_cols]
        targ_labels = [c for c in s_targets.columns.tolist()]

        train_size = int(s_features.shape[0] * train_frac)
        if train_features is None:
            train_features = s_features.values[:train_size]
            train_targets = s_targets.values[:train_size]
            test_features = s_features.values[train_size:]
            test_targets = s_targets.values[train_size:]
        else:
            train_features = np.vstack((train_features, s_features[:train_size]))
            train_targets = np.vstack((train_targets, s_targets[:train_size]))
            test_features = np.vstack((test_features, s_features[train_size:]))
            test_targets = np.vstack((test_targets, s_targets[train_size:]))

    return train_features, test_features, train_targets, test_targets, feat_labels, targ_labels

if __name__ == "__main__":
    top_stocks = get_top_stocks(num=6)
    dfs, _, _ = dp.load_stocks(stocks=top_stocks,
                               finra_shorts=False,
                               short_interest=False,
                               earliest_date=None,
                               calc_scores=False)

    latest_date, ld_df = get_latest_earliest_date(dfs)
    for s in dfs.keys():
        # standardize to latest date so they all have the same earliest time
        dfs[s] = dfs[s][dfs[s].index > latest_date]

    drop_cols(dfs)

    make_ohlcv_feats_targs_multithread(dfs, threshold=0.01)
    train_features, test_features, train_targets, test_targets = get_features_targets_deep(dfs)
