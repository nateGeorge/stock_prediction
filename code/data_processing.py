"""
for loading and processing stockdata
"""
# custom
import dl_quandl_EOD as dlq
import calculate_ta_signals as cts
sys.path.append('/home/nate/github/scrape_stocks/')
import scrape_finra_shorts as sfs


def load_stocks(stocks=['NAVI', 'EXAS'], TAs=True, finra_shorts=True, short_interest=True):
    """
    :param stocks: list of strings; tickers (must be caps)
    :param TAs: boolean, if true, calculates technical indicators
    :param shorts: boolean, if true, adds all short data

    :returns: dict of pandas dataframes with tickers as keys
    """
    dfs = dlq.load_stocks(stocks=stocks)
    if TAs:
        for s in stocks:
            cts.create_tas(dfs[s])

    if finra_shorts:
        finra_shorts = sfs.load_all_data()
