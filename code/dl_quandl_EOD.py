# core
import io
import os
import re
import time
import zipfile
import datetime
import glob
import gc

# installed
import quandl
import pandas as pd
import requests as req
from pytz import timezone
from concurrent.futures import ProcessPoolExecutor
import pandas_market_calendars as mcal
import pytz

# custom
from utils import get_home_dir


DEFAULT_STORAGE = '/home/nate/eod_data/'
# get todays date for checking if files up-to-date
MTN = timezone('America/Denver')
TODAY = datetime.datetime.now(MTN)
WEEKDAY = TODAY.weekday()
HOUR = TODAY.hour

HOME_DIR = get_home_dir()
HEADERS = ['Ticker',
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

Q_KEY = os.environ.get('quandl_api')
STOCKLIST = "../stockdata/goldstocks.txt"

quandl.ApiConfig.api_key = Q_KEY

def get_stocklist():
    """
    """
    url = 'http://static.quandl.com/end_of_day_us_stocks/ticker_list.csv'
    df = pd.read_csv(url)
    return df


def download_all_stocks_fast_csv(write_csv=False):
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


def download_entire_db(storage_path=DEFAULT_STORAGE,
                        remove_last=True,
                        return_df=False,
                        return_latest_date=False):
    """
    downloads entire database and saves to .h5, replacing old file
    :param storage_path: string, temporary location where to save the full csv file
    :param remove_last: removes last instance of the EOD dataset
    """
    # first check if we have the latest data
    zip_file_url = 'https://www.quandl.com/api/v3/databases/EOD/data?api_key=' + Q_KEY
    r = req.get(zip_file_url)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall(path=storage_path)
    df = pd.read_csv(storage_path + \
                    z.filelist[0].filename,
                    names=HEADERS,
                    index_col=1,
                    parse_dates=True,
                    infer_datetime_format=True)
    latest_date = df.index.max().date().strftime('%Y%m%d')
    df.to_hdf(storage_path + 'EOD_' + latest_date + '.h5',
                key='data',
                complib='blosc',
                complevel=9)
    if remove_last:
        files = glob.glob(storage_path + 'EOD_*.h5')
        files = [f for f in files if len(f.split('/')[-1]) == 15]  # don't want any of the small files, only full DBs
        print(sorted(files, key=os.path.getctime))
        if len(files) != 1:
            latest_file = sorted(files, key=os.path.getctime)[-2]
            print('removing', latest_file)
            os.remove(latest_file)

    os.remove(storage_path + z.filelist[0].filename)

    if return_df:
        return df
    elif return_latest_date:
        return pd.to_datetime(df.index.max().date())


def check_market_status():
    """
    Checks to see if market is open today.
    Uses the pandas_market_calendars package as mcal
    """
    # today = datetime.datetime.now(pytz.timezone('America/New_York')).date()
    today_utc = pd.to_datetime('now').date()
    ndq = mcal.get_calendar('NASDAQ')
    open_days = ndq.schedule(start_date=today_utc - pd.Timedelta('10 days'), end_date=today_utc)
    if today_utc in open_days.index:
        return open_days
    else:
        return None


def daily_download_entire_db(storage_path=DEFAULT_STORAGE):
    """
    checks if it is a trading day today, and downloads entire db after it has been updated
    (930pm ET)
    need to refactor -- this is messy and touchy.  Have to start before midnight UTC
    to work ideally
    """
    latest_db_date = get_latest_db_date()
    while True:
        latest_close_date = get_latest_close_date()
        today_utc = pd.to_datetime('now')
        today_ny = datetime.datetime.now(pytz.timezone('America/New_York'))
        pd_today_ny = pd.to_datetime(today_ny.date())
        if latest_db_date.date() != latest_close_date.date():
            if (latest_close_date.date() - latest_db_date.date()) >= pd.Timedelta('1D'):
                if today_utc.hour > latest_close_date.hour:
                    print('db more than 1 day out of date, downloading...')
                    latest_db_date = download_entire_db(return_latest_date=True)
            elif pd_today_ny.date() == latest_close_date.date():  # if the market is open and the db isn't up to date with today...
                if today_ny.hour >= 22:
                    print('downloading db with update from today...')
                    latest_db_date = download_entire_db(return_latest_date=True)

        print('sleeping 1h...')
        time.sleep(3600)

        # old code...don't think I need this anymore
        #     open_days = check_market_status()
        #     if open_days is not None:
        #         close_date = open_days.loc[today_utc.date()]['market_close']
        #         # TODO: add check if after closing time
        #         if today_utc.dayofyear > close_date.dayofyear or today_utc.year > close_date.year:
        #             if today_ny.hour > 10:  # need to wait until it has been processed to download
        #                 last_scrape = today_ny.date()
        #                 print('downloading db...')
        #                 download_entire_db()
        #         else:
        #             # need to make it wait number of hours until close
        #             print('waiting for market to close, waiting 1 hour...')
        #             time.sleep(3600)
        #     else:
        #         # need to wait till market will be open then closed next
        #         print('market closed today, waiting 1 hour...')
        #         time.sleep(3600)  # wait 1 hour
        # else:
        #     # need to make this more intelligent so it waits until the next day
        #     print('already scraped today, waiting 1 hour to check again...')
        #     time.sleep(3600)


def get_latest_db_date(storage_path=DEFAULT_STORAGE):
    """
    gets the date of the last full scrape of the db
    """
    files = glob.glob(storage_path + 'EOD_*.h5')
    if len(files) > 0:
        files = [f for f in files if len(f.split('/')[-1]) == 15]  # don't want any of the small files, only full DBs
        latest_file = sorted(files, key=os.path.getctime)[-1]
        last_date = pd.to_datetime(latest_file[-11:-3])
        return last_date

    return None


def get_latest_close_date(market='NASDAQ'):
    """
    gets the latest date the markets were open (NASDAQ), and returns the closing datetime
    """
    # today = datetime.datetime.now(pytz.timezone('America/New_York')).date()
    today_utc = pd.to_datetime('now').date()
    ndq = mcal.get_calendar(market)
    open_days = ndq.schedule(start_date=today_utc - pd.Timedelta('10 days'), end_date=today_utc)
    return open_days.iloc[-1]['market_close']


def check_market_status():
    """
    Checks to see if market is open today.
    Uses the pandas_market_calendars package as mcal
    """
    # today = datetime.datetime.now(pytz.timezone('America/New_York')).date()
    today_utc = pd.to_datetime('now').date()
    ndq = mcal.get_calendar('NASDAQ')
    open_days = ndq.schedule(start_date=today_utc - pd.Timedelta('10 days'), end_date=today_utc)
    if today_utc in open_days.index:
        return open_days
    else:
        return None


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


def load_one_stock_hdf(filename):
    df = pd.read_hdf(filename, index_col=0, parse_dates=True)
    return df


def load_one_stock_fulldf(full_df, s, make_files, filename):
    # don't think I need that stuff anymore
    df = full_df[full_df['Ticker'] == s]
    # df.loc[:, 'Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
    # df.set_index('Date', inplace=True)
    if make_files:
        df.to_hdf(filename, key='data', comlib='blosc', complevel=9)

    return df


def get_latest_eod(storage_path=DEFAULT_STORAGE, return_file=False):
    files = glob.glob(storage_path + '*.h5')
    latest_file = sorted(files, key=os.path.getctime)[-1]
    latest_eod = latest_file[-11:-3]
    if return_file:
        return latest_file

    return latest_eod


def make_small_df(storage_path=DEFAULT_STORAGE,
                filename='EOD_{}.h5',
                latest_eod=None,
                earliest_date='20150101'):
    """
    makes smaller h5 file with only data after specific time
    only have historic data for last 2 years for shortsqueeze, so that's an example
    :param latest_eod: string, YYYYMMDD
    :param earliest_date: string, format YYYYmmdd, earliest date to keep in df
    """
    if latest_eod is None:
        latest_eod = get_latest_eod()

    eod_datapath = storage_path + filename.format(latest_eod)
    new_filename = storage_path + filename.format(earliest_date + '_' + latest_eod)
    full_df = pd.read_hdf(eod_datapath, names=HEADERS)
    # hmm seemed to used to need this, not anymore
    # full_df['Date'] = pd.to_datetime(full_df['Date'], format='%Y-%m-%d')
    full_df = full_df[full_df.index > pd.to_datetime(earliest_date, format='%Y%m%d')]
    full_df.to_hdf(new_filename, key='data', complib='blosc', complevel=9)


def load_stocks(datapath=HOME_DIR + 'stockdata/',
                make_files=False,
                eod_datapath=DEFAULT_STORAGE,
                eod_filename='EOD_{}.h5',
                latest_eod=None,
                verbose=False,
                earliest_date=None):
    """
    :param datapath: string; path to stock datafiles
    :param make_files: bool, will save individual stock file if true (loading full dataset is quite slow)
    :param eod_datapath: string, path to full eod data
    :param latest_eod: string, yyyymmdd; latest day eod data was collected

    :returns: dictionary of dataframes with tickers as keys
    """
    if latest_eod is None:
        latest_eod = get_latest_eod()

    if earliest_date is None:
        eod_datapath = eod_datapath + eod_filename.format(latest_eod)
    else:
        eod_datapath = eod_datapath + eod_filename.format(earliest_date + '_' + latest_eod)
        if not os.path.exists(eod_datapath):
            make_small_df(earliest_date=earliest_date)

    dfs = {}
    # load big df with everything, and load all stocks
    full_df = pd.read_hdf(eod_datapath, names=HEADERS)
    tickers = set(full_df['Ticker'])
    stk_grps = full_df.groupby(by='Ticker')
    for t in tickers:
        dfs[t] = stk_grps.get_group(t)

    # was doing this before but I think I don't have to...
    # jobs = []
    # with ProcessPoolExecutor() as executor:
    #     for s in stocks:
    #         if s in tickers:
    #             filename = datapath + s + '_{}.h5'.format(latest_eod)
    #             if os.path.exists(filename):
    #                 r = executor.submit(load_one_stock_hdf, filename)
    #                 jobs.append((s, r))
    #             else:
    #                 if verbose:
    #                     print('loading', s)
    #                 r = executor.submit(load_one_stock_fulldf,
    #                                     full_df,
    #                                     s,
    #                                     make_files,
    #                                     filename)
    #         else:
    #             if verbose:
    #                 print(s, 'not in tickers')
    #             continue
    #
    # for s, r in jobs:
    #     res = r.result()
    #     if res is not None:
    #         dfs[s] = res
    #     else:
    #         print('result was None for', s)
    #
    # del jobs
    # gc.collect()

    if len(dfs) == 0:
        print('WARNING: no stocks were in the data, returning None')
        return None

    return dfs


def convert_full_df_to_hdf(eod_datapath=DEFAULT_STORAGE + 'EOD_{}.csv', latest_eod='20170812'):
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
