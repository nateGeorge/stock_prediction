import pandas as pd
import numpy as np


def create_new_features(df):
    """
    Creates features for price differences:
    high-low
    close-open
    """
    df['close-open'] = df['Adj_Close'] - df['Adj_Open']
    df['high-low'] = df['Adj_High'] - df['Adj_Low']
    df['close-open_pct'] = df['close-open'] / df['Adj_Close'] * 100
    df['high-low_pct'] = df['high-low'] / df['Adj_Close'] * 100

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
    for s in dfs.keys():
        data_points = dfs[s].shape[0]
        # create time-lagged features
        features = []
        targets = []
        pred_dates = dfs[s].iloc[history_days + future_days:]  # dates of the prediction days
        for i in range(history_days, data_points - future_days):
            features.append(dfs[s].iloc[i - history_days:i + 1][['Adj_Open', 'Adj_High', 'Adj_Low', 'Adj_Close', 'Adj_Volume']].values.ravel())
            targets.append(dfs[s].iloc[i + future_days]['Adj_Close'])

        feats[s] = np.array(features)
        targs[s] = np.array(targets)

    return feats, targs, pred_dates


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
