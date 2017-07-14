import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler as SS


def create_new_features(df):
    """
    Creates features for price differences:
    high-low
    close-open
    """
    df['close-open'] = df['Adj_Close'] - df['Adj_Open']
    df['high-low'] = df['Adj_High'] - df['Adj_Low']

    # create pct features
    df['close-open_pct'] = df['close-open'] / df['Adj_Close'] * 100
    df['high-low_pct'] = df['high-low'] / df['Adj_Close'] * 100

    df['close-close_pct'] = df['Adj_Close'].diff(periods=1) / df['Adj_Close'] * 100
    # avoid nan as first entry
    df['close-close_pct'].iloc[0] = df['close-close_pct'].iloc[1]

    df['open-open_pct'] = df['Adj_Open'].diff(periods=1) / df['Adj_Open'] * 100
    df['open-open_pct'].iloc[0] = df['open-open_pct'].iloc[1]
    df['high-high_pct'] = df['Adj_High'].diff(periods=1) / df['Adj_High'] * 100
    df['high-high_pct'].iloc[0] = df['high-high_pct'].iloc[1]
    df['low-low_pct'] = df['Adj_Low'].diff(periods=1) / df['Adj_Low'] * 100
    df['low-low_pct'].iloc[0] = df['low-low_pct'].iloc[1]
    df['vol-vol_pct'] = df['Adj_Volume'].diff(periods=1) / df['Adj_Volume'] * 100
    df['vol-vol_pct'].iloc[0] = df['vol-vol_pct'].iloc[1]

    return df


def create_hist_feats(dfs, history_days=30, future_days=5):
    """
    Creates features from historical data.  Assumes dataframe goes from
    oldest date at .iloc[0] to newest date at .iloc[-1]
    :param history_days number of days to use for prediction:
    :param future_days days out in the future we want to predict for
    """
    feats = {}
    targs = {}
    dates = {}
    for s in dfs.keys():
        data_points = dfs[s].shape[0]
        # create time-lagged features
        features = []
        targets = []
        pred_dates = dfs[s].iloc[history_days + future_days:].index  # dates of the prediction days
        for i in range(history_days, data_points - future_days):
            features.append(dfs[s].iloc[i - history_days:i + 1][['Adj_Open', 'Adj_High', 'Adj_Low', 'Adj_Close', 'Adj_Volume']].values.ravel())
            targets.append(dfs[s].iloc[i + future_days]['Adj_Close'])

        feats[s] = np.array(features)
        targs[s] = np.array(targets)
        dates[s] = pred_dates

    return feats, targs, dates


def create_hist_feats_all(dfs, history_days=30, future_days=5):
    """
    Creates features from historical data, but creates features up to the most
    current date available.  Assumes dataframe goes from
    oldest date at .iloc[0] to newest date at .iloc[-1]
    :param history_days number of days to use for prediction:
    :param future_days days out in the future we want to predict for
    """
    feats = {}
    targs = {}
    for s in dfs.keys():
        data_points = dfs[s].shape[0]
        # create time-lagged features
        features = []
        for i in range(history_days, data_points):
            features.append(dfs[s].iloc[i - history_days:i + 1][['Adj_Open', 'Adj_High', 'Adj_Low', 'Adj_Close', 'Adj_Volume']].values.ravel())

        feats[s] = np.array(features)

    return feats


def create_hist_feats_pct(dfs, history_days=120, future_days=20):
    feats = {}
    targs = {}
    dates = {}
    for s in dfs.keys():
        data_points = dfs[s].shape[0]
        # create time-lagged features
        features = []
        targets = []
        pred_dates = dfs[s].iloc[history_days + future_days:].index  # dates of the prediction days
        feat_cols = ['close-open_pct_scaled',
                'high-low_pct_scaled',
                'close-close_pct_scaled',
                'open-open_pct_scaled',
                'high-high_pct_scaled',
                'low-low_pct_scaled',
                'vol-vol_pct_scaled']
        for i in range(history_days, data_points - future_days):
            features.append(dfs[s].iloc[i - history_days:i + 1][feat_cols].values.ravel())
            targets.append(dfs[s].iloc[i + future_days]['close-close_pct_scaled'])

        feats[s] = np.array(features)
        targs[s] = np.array(targets)
        dates[s] = pred_dates

    return feats, targs, dates


def scale_pcts(df):
    feats = ['close-open_pct',
            'high-low_pct',
            'close-close_pct',
            'open-open_pct',
            'high-high_pct',
            'low-low_pct',
            'vol-vol_pct']

    scalers = []
    for f in feats:
        sc = SS()
        df[f + '_scaled'] = sc.fit_transform(df[f].values.reshape(-1, 1))
        scalers.append(sc)

    return df, scalers


def unscale_pct_close_preds(preds, scalers, days=20):
    resc = []
    cc_sc = scalers[2]
    for i in range(df.iloc[-days:].shape[0]):
        pct = preds.iloc[-days + i]
        unsc = cc_sc.inverse_transform(pct.reshape(-1, 1))[0][0]
        resc.append(unsc)

    resc = np.array(resc)

    return resc


def train_test_split_pcts(feats, targs, train_frac=0.85):
    samples = targs.shape[0]
    train_size = int(train_frac * samples)
    train_fs = feats[:train_size, :]
    test_fs = feats[train_size:, :]
    train_ts = targs[:train_size].reshape(-1, 1)
    test_ts = targs[train_size:].reshape(-1, 1)

    return train_fs, test_fs, train_ts, test_ts
