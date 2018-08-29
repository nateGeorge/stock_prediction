import sys
sys.path.append('../../scrape_ib')
import scrape_ib

import calculate_ta_signals as cts

tsla_3min = scrape_ib.load_data('TSLA')

# updates df in place
cts.create_tas(tsla_3min, ohlcv_cols=['open', 'high', 'low', 'close', 'volume'], tp=False)


# create future price change
tsla_3min['60m_future_price'] = tsla_3min['close'].shift(-20)
tsla_3min['60m_future_price_chg_pct'] = tsla_3min['60m_future_price'].pct_change(20)

import matplotlib.pyplot as plt
import seaborn as sns
f = plt.figure(figsize=(20, 20))
sns.heatmap(tsla_3min.corr())

tsla_3min.corr()['60m_future_price_chg_pct'][tsla_3min.corr()['60m_future_price_chg_pct'] > 0.1]

col = 'ht_dcp_cl'
plt.scatter(tsla_3min[col], tsla_3min['60m_future_price_chg_pct'])

# create range of past price changes to look for correlations
chg_cols = []
for i in range(1, 20):
    col = str(i * 3) + 'm_price_chg_pct'
    chg_cols.append(col)
    tsla_3min[col] = tsla_3min['close'].pct_change(i)

sns.heatmap(tsla_3min[chg_cols + ['60m_future_price_chg_pct']].corr())


keep_features = ['opt_vol_close',
                'rsi_cl',
                'ema_cl_diff',
                'mdm',  # volume feature
                'ht_dcp_cl',
                'high',
                'low',
                'close']


nona = tsla_3min.dropna()
feats = nona[keep_features]
targs = nona['60m_future_price_chg_pct']


# attempt with TSLA from 8-10 to 8-20 2018
trainsize = 0.8
train_idx = int(trainsize * feats.shape[0])
tr_feats = feats[:train_idx]
tr_targs = targs[:train_idx]
te_feats = feats[train_idx:]
te_targs = targs[train_idx:]


from pandas.plotting import autocorrelation_plot
autocorrelation_plot(tsla_3min['close'])
# WHY YOU NO WORK???!?!?
autocorrelation_plot(tsla_3min['15m_price_chg_pct'])
autocorrelation_plot(tsla_3min['30m_price_chg_pct'])
for i in range(1, 20):
    print(i)
    print(tsla_3min[str(i * 3) + 'm_price_chg_pct'].autocorr(i))

plt.scatter()

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import ParameterGrid

# check feature importances



grid = ParameterGrid({'n_estimators': [500],
                        'random_state': [42],
                        'max_depth': [3, 5, 7],
                        'max_features': ['auto', 2, 5],
                        'n_jobs': [-1],
                        'min_samples_split': [2, 5, 10]})

for g in grid:
    print(g)
    rfr = RandomForestRegressor(**g)
    rfr.fit(tr_feats, tr_targs)
    print(rfr.score(tr_feats, tr_targs))
    print(rfr.score(te_feats, te_targs))

best = {'max_features': 2, 'n_estimators': 500, 'max_depth': 3, 'min_samples_split': 5, 'random_state': 42, 'n_jobs': -1}

rfr = RandomForestRegressor(**best)
rfr.fit(tr_feats, tr_targs)
print(rfr.score(tr_feats, tr_targs))
print(rfr.score(te_feats, te_targs))
plt.scatter(rfr.predict(tr_feats), tr_targs, label='train')
plt.scatter(rfr.predict(te_feats), te_targs, label='test')

# best scores about 0.5 and -0.5
