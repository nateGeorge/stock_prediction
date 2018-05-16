import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# custom
import data_processing as dp

dfs, sh_int, fin_sh = dp.load_stocks(stocks=None, TAs=False, finra_shorts=False, short_interest=False, earlist_date=None)

# full_df = pd.concat([dfs[s] for s in dfs.keys()])
stocks = ['LNG', 'CHK', 'AMD']
small_df = pd.concat(dfs[s] for s in stocks)

abbrev_df = small_df[['Ticker', 'Adj_Close']]
table = abbrev_df.pivot(columns='Ticker')
table_monthly = table.resample('MS').first()#, closed='left')

# daily returns of stocks
returns_daily = table.pct_change()
# calculate monthly returns of the stocks
returns_monthly = table_monthly.pct_change()

# calculate monthly moving average of stocks
ewma_daily = returns_daily.ewm(span=30).mean()
ewma_monthly = ewma_daily.resample('MS').first()


# daily covariance of stocks (for each monthly period)
covariances = {}
for i in returns_monthly.index:
    rtd_idx = returns_daily.index
    mask = (rtd_idx.month == i.month) & (rtd_idx.year == i.year)
    covariances[i] = returns_daily[mask].cov()



# empty dictionaries to store returns, volatility and weights of imiginary portfolios
port_returns = {}
port_volatility = {}
stock_weights = {}
sharpe_ratio = {}
max_sharpe = {}

# set the number of combinations for imaginary portfolios
num_assets = len(stocks)
num_portfolios = 5000

# get portfolio performances at each month
# populate the empty lists with each portfolios returns,risk and weights
for date in covariances.keys():
    print(date)
    cov = covariances[date]
    for single_portfolio in range(num_portfolios):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        returns = np.dot(weights, returns_monthly.loc[date])
        volatility = np.sqrt(np.dot(weights.T, np.dot(cov, weights)))
        port_returns.setdefault(date, []).append(returns)
        port_volatility.setdefault(date, []).append(volatility)
        stock_weights.setdefault(date, []).append(weights)
        sharpe = returns / volatility
        sharpe_ratio.setdefault(date, []).append(sharpe)

    max_sharpe[date] = np.argmax(sharpe_ratio[date])

# make features and targets
targets = []
features = []

for date in covariances.keys():
    best_idx = max_sharpe[date]
    targets.append(stock_weights[date][best_idx])
    features.append(ewma_monthly.loc[date].values)

targets = np.array(targets)
features = np.array(features)

feat_dict = {'feature_' + str(i): features[:, i] for i in range(features.shape[1])}
targ_dict = {'target_' + str(i): targets[:, i] for i in range(targets.shape[1])}

feat_targ_df = pd.DataFrame({**feat_dict, **targ_dict})

for date in covariances.keys():
    best_idx = max_sharpe[date]
    print(port_returns[date][best_idx])


def plot_frontier(date, stocks):
    # date = list(covariances.keys())[1]
    portfolio = {'Returns': port_returns[date],
                 'Volatility': port_volatility[date],
                 'Sharpe Ratio': sharpe_ratio[date]}

    # extend original dictionary to accomodate each ticker and weight in the portfolio
    for i, symbol in enumerate(stocks):
        portfolio[symbol + ' Weight'] = [Weight[i] for Weight in stock_weights[date]]

    # make a nice dataframe of the extended dictionary
    df = pd.DataFrame(portfolio)

    # get better labels for desired arrangement of columns
    column_order = ['Returns', 'Volatility', 'Sharpe Ratio'] + [stock+' Weight' for stock in stocks]

    # reorder dataframe columns
    df = df[column_order]

    # plot frontier, max sharpe & min Volatility values with a scatterplot
    plt.style.use('seaborn-dark')
    df.plot.scatter(x='Volatility', y='Returns', c='Sharpe Ratio',
                    cmap='RdYlGn', edgecolors='black', figsize=(10, 8), grid=True)
    plt.xlabel('Volatility (Std. Deviation)')
    plt.ylabel('Expected Returns')
    plt.title('Efficient Frontier')
    plt.show()
