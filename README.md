# stock_prediction
A bash script to scrape yahoo stocks.  Machine learning in Python/tensorflow to predict future prices.

This repo is super messy.  It has a lot of different functions in it.  I ended up primarily using it to save Quandl EOD data daily, and to load that data.

# Yahoo hidden API
Part of this used to use the hidden Yahoo API, which can be found [here](https://greenido.wordpress.com/2009/12/22/work-like-a-pro-with-yahoo-finance-hidden-api/).
See [this](http://luminouslogic.com/how-to-normalize-historical-data-for-splits-dividends-etc.htm) for how to adjust for splits.

Now there is the yfinance Python package that seems to work for this purpose.  Backtrader also has a way to access Yahoo finance data.
