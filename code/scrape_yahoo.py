import datetime
from pytz import timezone
import requests as req
import pandas as pd
import numpy as np
from collections import OrderedDict

MTN = timezone('America/Denver')

TODAY = datetime.datetime.now(MTN)
YEAR = str(TODAY.year)
MONTH = str(TODAY.month).zfill(2)
DAY = str(TODAY.day).zfill(2)

HIST_BASE_URL = "https://ichart.yahoo.com/table.csv?"
QUOTE_BASE_URL = "https://download.finance.yahoo.com/d/quotes.csv?"

STOCKLIST = "../stockdata/goldstocks.txt"

def download_stocks(stocklist=STOCKLIST):
    # load stocklist
    with open(stocklist) as f:
        stocks = f.read().strip('\n').split('\n')

    # download each stock
    for s in stocks:
        scrape_url = HIST_BASE_URL + 's=' + s + '&a=0&b=0&c=0&d=' + MONTH + '&e=' + DAY + '&f=' + YEAR
        print(scrape_url)
        res = req.get(scrape_url)
        if not res.ok:
            print('problem! : ' + str(res.status_code))
            continue

        test = res.content.strip(b'\n').split(b'\n')
        labels = test[0].decode('ascii').split(',')
        datadict = OrderedDict()
        for line in test[1:]:
            data = line.decode('ascii').split(',')
            datadict.setdefault(labels[0], []).append(datetime.datetime.strptime(data[0], '%Y-%m-%d'))
            for d, l in zip(data[1:], labels[1:]):
                datadict.setdefault(l, []).append(float(d))

        df = pd.DataFrame(datadict)
        df.to_csv('../stockdata/' + s + '.csv.gz', index=False, compression='gzip')


def load_stocks(stocks=['GLD', 'DUST', 'NUGT']):
    dfs = {}
    for s in stocks:
        df = pd.read_csv('../stockdata/' + s + '.csv.gz', index_col=0, parse_dates=True)
        dfs[s] = df

    return dfs


def normalize_prices(df):
    """
    Currently, this normalizes the prices to the current actual price.  The
    'adjusted close' is the close adjusted for all splits, so it would
    be the close adjusted to match what it would be if there were no splits.
    http://luminouslogic.com/how-to-normalize-historical-data-for-splits-dividends-etc.htm
    """
    new_df = df.copy()
    price_cols = ['Open', 'Close', 'High', 'Low']
    # .iloc[0] is the latest date
    # usually this should be 1
    last_ratio = df.iloc[0]['Adj Close'] / df.iloc[0]['Close']
    # this is what will really normalize the data.  The adjusted close is
    # what the value would be if things were equalized for the latest data
    ratio = df['Adj Close'] / df['Close']
    for p in price_cols:
        new_df[p] = df[p] * ratio * last_ratio

    return new_df


def create_new_features(df):
    """
    Creates features for price differences:
    high-low
    close-open
    """
    df['close-open'] = df['Close'] - df['Open']
    df['high-low'] = df['High'] - df['Low']
    df['close-open_pct'] = df['close-open'] / df['Adj Close'] * 100
    df['high-low_pct'] = df['high-low'] / df['Adj Close'] * 100

    return df


def load_norm_crt_feat(stocks=['NUGT', 'DUST', 'GLD']):
    """
    Loads data, creates new features, and normalizes prices for splits.
    """
    dfs = load_stocks(stocks=stocks)
    for s in stocks:
        dfs[s] = create_new_features(dfs[s])
        dfs[s] = normalize_prices(dfs[s])

    return dfs


def create_hist_feats(dfs, history_days=30, future_days=5):
    """
    Creates features from historical data.
    :param history_days number of days to use for prediction:
    :param future_days days out in the future we want to predict for
    """
    feats = {}
    targs = {}
    for s in dfs.keys():
        data_points = dfs[s].shape[0]
        dfs[s] = dfs[s].iloc[::-1]  # reverses dataframe
        # create time-lagged features
        features = []
        targets = []
        for i in range(history_days, data_points - future_days):
            features.append(dfs[s].iloc[i - history_days:i][['Open', 'High', 'Low', 'Close', 'Volume']].values.ravel())
            targets.append(dfs[s].iloc[i + future_days]['Close'])

        feats[s] = np.array(features)
        targs[s] = np.array(targets)

    return feats, targs


if __name__ == "__main__":
    download_stocks()
