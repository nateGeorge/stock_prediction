import datetime
from pytz import timezone
import requests as req
import pandas as pd
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
    last_ratio = df.iloc[0]['Close'] / df.iloc[0]['Adj Close']
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


if __name__ == "__main__":
    download_stocks()
