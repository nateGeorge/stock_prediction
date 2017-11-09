# core
import io
import os
import re
import time
import zipfile
import datetime
import requests as req
from pytz import timezone

# installed
import quandl
import pandas as pd

# custom
from utils import get_home_dir

# get todays date for checking if files up-to-date
MTN = timezone('America/Denver')
TODAY = datetime.datetime.now(MTN)
WEEKDAY = TODAY.weekday()
HOUR = TODAY.hour

HOME_DIR = get_home_dir()

Q_KEY = os.environ.get('quandl_api')
STOCKLIST = "../stockdata/goldstocks.txt"

quandl.ApiConfig.api_key = Q_KEY

def get_stocklist():
    """
    """
    url = 'http://static.quandl.com/end_of_day_us_stocks/ticker_list.csv'
    df = pd.read_csv(url)
    return df


def download_all_stocks_fast(write_csv=False):
    """
    """
    zip_file_url = 'https://www.quandl.com/api/v3/databases/EOD/data?api_key=' + Q_KEY
    r = req.get(zip_file_url)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall(path='../stockdata/')
    headers = update_all_stocks(return_headers=True)
    df = pd.read_csv('../stockdata/' + \
                    z.filelist[0].filename,
                    names=headers,
                    index_col=1,
                    parse_dates=True)
    df.sort_index(inplace=True)
    if write_csv:
        # compression really slows it down...don't recommend
        df.to_csv('../stockdata/all_stocks.csv.gzip', compression='gzip')
        os.remove('../stockdata/' + z.filelist[0].filename)

    return df


def update_all_stocks(return_headers=False, update_small_file=False):
    """
    return_headers will just return the column names.
    update_small_file will just update the small file that starts on 1/1/2000
    """
    # 7-13-2017: 28788363 rows in full df
    zip_file_url = 'https://www.quandl.com/api/v3/databases/EOD/download?api_key=' + \
        Q_KEY + '&download_type=partial'
    r = req.get(zip_file_url)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall(path='../stockdata/')
    if return_headers:
        df = pd.read_csv('../stockdata/' + z.filelist[0].filename, parse_dates=True)
        df.set_index('Date', inplace=True)
        new_c = [re.sub('.\s', '_', c) for c in df.columns]
        return new_c

    df = pd.read_csv('../stockdata/' + z.filelist[0].filename)
    # it  won't parse dates when it reads...
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    # fix problem with . and _ in Adjusted cols
    new_c = [re.sub('.\s', '_', c) for c in df.columns]
    df.columns = new_c
    full_df = pd.read_csv('../stockdata/all_stocks.csv.gzip',
                    parse_dates=True,
                    compression='gzip',
                    index_col=0)
    if (full_df.columns == df.columns).mean() != 1:
        print('WARNING! Columns in the full df do not match the updated df columns.')
        print('full_df cols:')
        print(full_df.columns)
        print('')
        print('update df cols:')
        print(df.columns)
        print('')
        print('aborting and returning current full_df')
        return full_df

    if df.index.max() > full_df.index.max():
        df.to_csv('../stockdata/all_stocks.csv.gzip', mode='a', compression='gzip')
        dtypes = ['object'] + ['float64'] * 10
        full_df = pd.read_csv('../stockdata/all_stocks.csv.gzip',
                                parse_dates=True,
                                compression='gzip',
                                index_col=0,
                                dtype=dtypes)

    os.remove('../stockdata/' + z.filelist[0].filename)

    return full_df


def get_last_n_days(n=100):
    """
    Retrieves and saves last n days from the full stock dataset for analysis.
    """
    df = pd.read_csv('../stockdata/all_stocks.csv.gzip', parse_dates=True)
    dates = sorted(df.index.unique())[-n:]
    new_df = df.loc[dates]
    new_df.to_csv('../stockdata/all_stocks_last' + str(n) + '_days.csv.gzip',
                    compression='gzip')
    return new_df


def download_all_stocks():
    """
    With about 8k stocks and about 2s per stock download, this would take forever.
    Don't use.
    """
    stocks = get_stocklist()
    dfs = {}
    for i, r in stocks.iterrows():
        start = time.time()
        s = r['Ticker']
        stockfile = '../stockdata/' + s + '.csv.gz'
        print('downloading', s)
        stock = quandl.get('EOD/' + s)
        stock.to_csv(stockfile, compression='gzip')
        dfs[s] = stock
        print('took', time.time() - start, 's')

    return dfs


def download_stocks(stocklist=STOCKLIST, fresh=False):
    """
    Downloads stock data and returns dict of pandas dataframes.
    First checks if data is up to date, if so, just loads the data.
    """
    # load stocklist
    with open(stocklist) as f:
        stocks = f.read().strip('\n').split('\n')

    dfs = {}
    for s in stocks:
        print(s)
        stockfile = '../stockdata/' + s + '.csv.gz'
        if fresh:
            print('downloading fresh')
            stock = quandl.get('EOD/' + s)
            stock.to_csv(stockfile, compression='gzip')
            dfs[s] = stock
            continue

        if os.path.exists(stockfile):
            stock = pd.read_csv(stockfile, index_col=0)
            stock.index = pd.to_datetime(stock.index)
            timedelta_step = 1
            if HOUR > 2 and WEEKDAY not in [5, 6]:  # for mtn time
                timedelta_step = 0
            elif WEEKDAY == 0:  # it's monday
                timedelta_step = 3  # can be up to last friday
            elif WEEKDAY in [5, 6]:  # if a weekend, last data is from friday
                timedelta_step = WEEKDAY - 4
            print('date gap:', TODAY.date() - stock.iloc[-2:].index[-1].date())
            print('step, timedelta:', timedelta_step, datetime.timedelta(timedelta_step))
            if (TODAY.date() - stock.iloc[-2:].index[-1].date()) <= datetime.timedelta(timedelta_step):
                dfs[s] = stock
                print('latest date close enough to up-to-date:')
                print(stock.iloc[-2:].index[-1].date())
                print('not downloading')
                print('')
                continue
            else:
                print('latest date is')
                print(stock.iloc[-2:].index[-1].date())
                print('downloading fresh')
                stock = quandl.get('EOD/' + s)
                stock.to_csv(stockfile, compression='gzip')
                dfs[s] = stock

    return dfs


def load_stocks(datapath=HOME_DIR + 'stockdata/',
                stocks=['GLD', 'DUST', 'NUGT'],
                make_files=True,
                eod_datapath='/home/nate/eod_data/EOD_{}.h5',
                latest_eod='20170812'):
    """
    :param datapath: string; path to stock datafiles
    :param stocks: list of strings, stock tickers (must be uppercase)
    :param make_files: bool, will save individual stock file if true (loading full dataset is quite slow)
    :param eod_datapath: string, path to full eod data
    :param latest_eod: string, yyyymmdd; latest day eod data was collected

    :returns: dictionary of dataframes with tickers as keys
    """
    eod_datapath = eod_datapath.format(latest_eod)
    dfs = {}
    full_df = None
    for s in stocks:
        filename = datapath + s + '_{}.h5'.format(latest_eod)
        if os.path.exists(filename):
            df = pd.read_hdf(filename, index_col=0, parse_dates=True)
        else:
            if full_df is None:
                headers = ['Ticker',
                           'Date',
                           'Open',
                           'High',
                           'Low',
                           'Close',
                           'Volume',
                           'Dividend',
                           'Split',
                           'Adj_Open',
                           'Adj_High',
                           'Adj_Low',
                           'Adj_Close',
                           'Adj_Volume']
                full_df = pd.read_hdf(eod_datapath, names=headers)
                tickers = set(full_df['Ticker'])

            if s in tickers:
                df = full_df[full_df['Ticker'] == s]
                df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
                df.set_index('Date', inplace=True)
                if make_files:
                    df.to_hdf(filename, key='data', comlib='blosc', complevel=9)
            else:
                print('stock not in tickers')

        dfs[s] = df

    return dfs


def convert_full_df_to_hdf(eod_datapath='/home/nate/eod_data/EOD_{}.csv', latest_eod='20170812'):
    eod_datapath = eod_datapath.format(latest_eod)
    eod_datapath_h5 = eod_datapath.strip('.csv')
    headers = ['Ticker',
               'Date',
               'Open',
               'High',
               'Low',
               'Close',
               'Volume',
               'Dividend',
               'Split',
               'Adj_Open',
               'Adj_High',
               'Adj_Low',
               'Adj_Close',
               'Adj_Volume']
    full_df = pd.read_csv(eod_datapath, names=headers)
    full_df.to_hdf(eod_datapath_h5, key='data', complib='blosc', complevel=9)
