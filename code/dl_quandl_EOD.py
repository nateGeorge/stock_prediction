import quandl
import os
import pandas as pd
import datetime
from pytz import timezone

# get todays date for checking if files up-to-date
MTN = timezone('America/Denver')
TODAY = datetime.datetime.now(MTN)
WEEKDAY = TODAY.weekday()

Q_KEY = os.environ.get('quandl_api')
STOCKLIST = "../stockdata/goldstocks.txt"


def download_stocks(stocklist=STOCKLIST, fresh=False):
    """
    Downloads stock data and returns dict of pandas dataframes.
    First checks if data is up to date, if so, just loads the data.
    """
    quandl.ApiConfig.api_key = Q_KEY
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
            if WEEKDAY == 0:  # it's monday
                timedelta_step = 3  # can be up to last friday
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


def load_stocks(stocks=['GLD', 'DUST', 'NUGT']):
    dfs = {}
    for s in stocks:
        df = pd.read_csv('../stockdata/' + s + '.csv.gz', index_col=0, parse_dates=True)
        dfs[s] = df

    return dfs
