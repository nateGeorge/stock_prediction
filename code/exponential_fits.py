"""
performs exponential fits on all stocks to find coherent trends


"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

import dl_quandl_EOD as dlq



def fit_exponential_curve(df, points=90, plot_fit=False):
    """
    from stocks_on_the_move_strategy github repo

    fits exponential curve to data

    df should be a dataframe from the 'stocks' dictionary of dataframes
    points is number of points to use in fit
    """
    # df = stocks['JBT']
    prices = df['Adj_Close'].iloc[-points:].values
    ln_prices = np.log(prices)
    X = np.arange(ln_prices.shape[0])#.reshape(-1, 1)  # reshape only needed for sklearn
    # the problem with a regular linear regression is the fit is biased towards smaller values,
    # which will be the older values
    # instead, we want to weight the points by their value
    # https://stackoverflow.com/a/3433503/4549682
    # http://mathworld.wolfram.com/LeastSquaresFittingExponential.html
    coefs = np.polyfit(X, ln_prices, 1, w=ln_prices)
    slope = np.exp(coefs[0])
    # equation: y = Ae^Bx
    preds = np.exp(coefs[1]) * np.exp(coefs[0] * X)
    # old way of doing it:
    # lr = LinearRegression()
    # lr.fit(X, ln_prices)
    # slope = lr.coef_[0]  # should be approximately how much in pct price changes per day
    # preds = lr.predict(X)

    annualized_slope = (slope) ** 250
    # r2 = lr.score(X, ln_prices)  # old way for sklearn model
    r2 = r2_score(prices, preds)
    rank_score = annualized_slope * r2

    if plot_fit:
        # note: this would need to be changed for sklearn fits
        # plot fit with ln(prices)
        plt.scatter(X, ln_prices)
        plt.plot(X, np.log(preds))
        plt.title('ln of prices')
        plt.show()

        # plot fit with normal prices
        plt.scatter(X, np.exp(ln_prices))
        plt.plot(X, preds)
        plt.title('normal prices')
        plt.show()

    return annualized_slope, r2, rank_score


class calc_fits:
    def __init__(self):
        self.slope = []
        self.annualized_slope = []
        self.r2 = []

    def fit_exponential_curve_series(self, series, points=4):
        """
        from stocks_on_the_move_strategy github repo

        fits exponential curve to data

        series is pandas series a prices
        points is number of points to use in fit
        """
        ln_prices = np.log(series)
        X = np.arange(ln_prices.shape[0])#.reshape(-1, 1)  # reshape only needed for sklearn
        # the problem with a regular linear regression is the fit is biased towards smaller values,
        # which will be the older values
        # instead, we want to weight the points by their value
        # https://stackoverflow.com/a/3433503/4549682
        # http://mathworld.wolfram.com/LeastSquaresFittingExponential.html
        coefs = np.polyfit(X, ln_prices, 1, w=ln_prices)
        slope = np.exp(coefs[0])
        # equation: y = Ae^Bx
        preds = np.exp(coefs[1]) * np.exp(coefs[0] * X)

        annualized_slope = (slope) ** 251
        # r2 = lr.score(X, ln_prices)  # old way for sklearn model
        r2 = r2_score(series, preds)
        rank_score = annualized_slope * r2

        self.slope.append(slope)
        self.annualized_slope.append(annualized_slope)
        self.r2.append(r2)
        return rank_score


    def fit_linear_series(self, series, points=4):
        """
        from stocks_on_the_move_strategy github repo

        fits linear fit to data

        series is pandas series a prices
        points is number of points to use in fit
        """
        X = np.arange(series.shape[0])
        # the problem with a regular linear regression is the fit is biased towards smaller values,
        # which will be the older values
        # instead, we want to weight the points by their value
        # https://stackoverflow.com/a/3433503/4549682
        # http://mathworld.wolfram.com/LeastSquaresFittingExponential.html
        coefs = np.polyfit(X, series, 1)#, w=ln_prices)
        slope = coefs[0]
        # equation: y = mx + b
        preds = coefs[0] * X + coefs[1]

        annualized_slope = np.sign(slope) * (slope) ** 252
        r2 = r2_score(series, preds)
        rank_score = annualized_slope * r2

        self.slope.append(slope)
        self.annualized_slope.append(annualized_slope)
        self.r2.append(r2)
        return rank_score


def get_fits(df, col='Adj_Close', n_s=[5, 10, 15, 20, 60]):
    """
    calculates linear fits to
    """
    # return multiple values from a rolling operation
    # https://stackoverflow.com/a/39064656/4549682
    # get slope from 4 days' points
    rank_score = pd.DataFrame()
    for n in n_s:
        calc_ = calc_fits()
        rank_score_temp = df[col].rolling(n).apply(lambda x: calc_.fit_linear_series(x, points=n), raw=True)
        rank_score_temp = rank_score_temp.to_frame()
        rank_score_temp.columns = ['{}d_rank_score'.format(n)]
        nan_pad = [np.nan] * (n - 1)
        rank_score_temp['{}d_r2'.format(n)] = nan_pad + calc_.r2
        rank_score_temp['{}d_ann_slope'.format(n)] = nan_pad + calc_.annualized_slope
        rank_score_temp['{}d_slope'.format(n)] = nan_pad + calc_.slope
        rank_score = pd.concat([rank_score, rank_score_temp], axis=1)  # concats along columns

    rank_score['Adj_Close'] = df['Adj_Close']
    rank_score['close_pct_chg'] = rank_score[col].pct_change()
    rank_score = get_future_pct_chgs(rank_score)
    rank_score = get_autocorrs(rank_score)

    return rank_score


def get_future_pct_chgs(df):
    """
    from the 'close_pct_chg' column, this gets the close pct change in the future
    for a variety of days
    """
    for i in list(range(1, 6)) + [10, 15, 20]:
        df['{}d_future_pct_chg'.format(i)] = df['close_pct_chg'].shift(-i)

    return df


def get_autocorrs(df):
    """
    gets autocorrelation trends for a variety of periods
    """
    for n in [5, 10, 15, 20, 60]:
        df['{}d_1d_autocorr'.format(n)] = df['Adj_Close'].rolling(n).apply(lambda x: pd.Series(x).autocorr(), raw=True)
        if n == 5:
            continue  # need at least n+2 for autocorr rolling

        df['{}d_5d_autocorr'.format(n)] = df['Adj_Close'].rolling(n).apply(lambda x: pd.Series(x).autocorr(5), raw=True)
        # minimum of n + 2 for autocorr rolling period (somehow)
        # df['5d_3d_autocorr'] = df['Adj_Close'].rolling(5).apply(lambda x: pd.Series(x).autocorr(3), raw=True)
        # df['6d_4d_autocorr'] = df['Adj_Close'].rolling(6).apply(lambda x: pd.Series(x).autocorr(4), raw=True)
    return df


if __name__ == "__main__":
    stocks = dlq.load_stocks()
    qqq = stocks['QQQ']
    tqqq = stocks['TQQQ']

    # test if can fit negative trends
    tqqq_abbrev = tqqq.loc[:'2018-3-23']
    ann_sl, r2, rs = fit_exponential_curve(tqqq_abbrev, 3, plot_fit=True)

    # c1 = pd.rolling_apply(tqqq['Adj_Close'], 7, lambda x: pd.Series(x).autocorr(1))

    # "trends" will be series of autocorrelations over a certain threshold
    # 4 days seems to be shortest possible -- 3 days is always +1 or -1
    c2 = qqq['Adj_Close'].rolling(4).apply(lambda x: pd.Series(x).autocorr(), raw=True)
    c2.hist()
    plt.show()
    c2.plot()
    plt.show()


    c2 = qqq['Adj_Close'].rolling(9).apply(lambda x: pd.Series(x).autocorr(), raw=True)
    c2.hist()
    plt.show()
    c2.plot()
    plt.show()

    rank_score = get_fits(qqq)
    # try ML model
    rank_score.corr()

    # make train/test
    feat_cols = ['10d_r2', '10d_slope', 'close_pct_chg', '5d_1d_autocorr', '10d_5d_autocorr']
    targ_cols = ['5d_future_pct_chg', '10d_future_pct_chg', '20d_future_pct_chg']
    nona = rank_score.dropna()
    train_idx = int(rank_score.shape[0] * 0.75)
    train_x = nona.iloc[:train_idx][feat_cols]
    train_y = nona.iloc[:train_idx][targ_cols]
    test_x = nona.iloc[train_idx:][feat_cols]
    test_y = nona.iloc[train_idx:][targ_cols]

    from sklearn.ensemble import RandomForestRegressor
    rfr = RandomForestRegressor(n_estimators=500, max_depth=5, random_state=42, n_jobs=-1)

    rfr.fit(train_x, train_y)
    print(rfr.score(train_x, train_y))
    print(rfr.score(test_x, test_y))
