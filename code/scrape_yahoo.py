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
        res = req.get(scrape_url)
        if not res.ok:
            print('problem! :' + res.status_code)
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


def load_stocks(stocks=['GLD']):
    dfs = {}
    for s in stocks:
        df = pd.read_csv('../stockdata/' + s + '.csv.gz', index_col=0, parse_dates=True)
        dfs[s] = df

    return dfs


if __name__ == "__main__":
    download_stocks()
