import numpy as np
import talib

def get_indicator_list():
    """
    returns list of indicators
    """
    return ['bband_u_cl', # bollinger bands
     'bband_m_cl',
     'bband_l_cl',
     'bband_u_tp',
     'bband_m_tp',
     'bband_l_tp',
     'bband_u_cl_diff',
     'bband_m_cl_diff',
     'bband_l_cl_diff',
     'bband_u_cl_diff_hi',
     'bband_l_cl_diff_lo',
     'bband_u_tp_diff',
     'bband_m_tp_diff',
     'bband_l_tp_diff',
     'bband_u_tp_diff_hi',
     'bband_l_tp_diff_lo',
     'dema_cl',
     'dema_tp',
     'dema_cl_diff',
     'dema_tp_diff',
     'ema_cl',
     'ema_tp',
     'ema_cl_diff',
     'ema_tp_diff',
     'ht_tl_cl',
     'ht_tl_tp',
     'ht_tl_cl_diff',
     'ht_tl_tp_diff',
     'kama_cl',
     'kama_tp',
     'kama_cl_diff',
     'kama_tp_diff',
    #  'mama_cl',  # having problems with these
    #  'mama_tp',
    #  'fama_cl',
    #  'fama_tp',
    #  'mama_cl_osc',
    #  'mama_tp_osc',
     'midp_cl',
     'midp_tp',
     'midp_cl_diff',
     'midp_tp_diff',
     'midpr',
     'midpr_diff',
     'sar',
     'sar_diff',
     'sma_10_cl',
     'sma_10_tp',
     'sma_20_cl',
     'sma_20_tp',
     'sma_30_cl',
     'sma_30_tp',
     'sma_40_cl',
     'sma_40_tp',
     'tema_cl',
     'tema_tp',
     'tema_cl_diff',
     'tema_tp_diff',
     'trima_cl',
     'trima_tp',
     'trima_cl_diff',
     'trima_tp_diff',
     'wma_cl',
     'wma_tp',
     'wma_cl_diff',
     'wma_tp_diff',
     'adx',
     'adxr',
     'apo_cl',
     'apo_tp',
     'arup', # aroon
     'ardn',
     'aroonosc',
     'bop',
     'cci',
     'cmo_cl',
     'cmo_tp',
     'dx',
     'macd_cl',
     'macdsignal_cl',
     'macdhist_cl',
     'macd_tp',
     'macdsignal_tp',
     'macdhist_tp',
     'mfi',
     'mdi',
     'mdm',
     'mom_cl',
     'mom_tp',
     'pldi',
     'pldm',
     'ppo_cl',
     'ppo_tp',
     'roc_cl',
     'roc_tp',
     'rocp_cl',
     'rocp_tp',
     'rocr_cl',
     'rocr_tp',
     'rocr_cl_100',
     'rocr_tp_100',
     'rsi_cl',
     'rsi_tp',
     'slowk', # stochastic oscillator
     'slowd',
     'fastk',
     'fastd',
     'strsi_cl_k',
     'strsi_cl_d',
     'strsi_tp_k',
     'strsi_tp_d',
     'trix_cl',
     'trix_tp',
     'ultosc',
     'willr',
     'ad',
     'adosc',
     'obv_cl',
     'obv_tp',
     'atr',
     'natr',
     'trange',
     'ht_dcp_cl',
     'ht_dcp_tp',
     'ht_dcph_cl',
     'ht_dcph_tp',
     'ht_ph_cl',
     'ht_ph_tp',
     'ht_q_cl',
     'ht_q_tp',
     'ht_s_cl',
     'ht_s_tp',
     'ht_ls_cl',
     'ht_ls_tp',
     'ht_tr_cl',
     'ht_tr_tp'
     ]


def create_tas(bars,
                verbose=False,
                ohlcv_cols=['High', 'Low', 'Open', 'Close', 'Volume'],
                return_df=False):
    """
    :param bars: resampled pandas dataframe with open, high, low, close, volume, and typical_price columns
    :param verbose: boolean, if true, prints more debug
    :param ohlcv_cols: list of strings; the column names for high, low, open, close, and volume

    :returns: pandas dataframe with TA signals calculated (modifies dataframe in place)
    """
    h, l, o, c, v = ohlcv_cols
    if 'typical_price' not in bars.columns:
        bars['typical_price'] = bars[['High', 'Low', 'Close']].mean(axis=1)

    # bollinger bands
    # strange bug, if values are small, need to multiply to larger value for some reason
    mult = 1
    last_close = bars.iloc[0][c]
    lc_m = last_close * mult
    while lc_m < 1:
        mult *= 10
        lc_m = last_close * mult

    if verbose:
        print('using multiplier of', mult)

    mult_tp = bars['typical_price'].values * mult
    mult_close = bars[c].values * mult
    mult_open = bars[o].values * mult
    mult_high = bars[h].values * mult
    mult_low = bars[l].values * mult


    ### overlap studies
    # bollinger bands -- should probably put these into another indicator
    upper_cl, middle_cl, lower_cl = talib.BBANDS(mult_close,
                                    timeperiod=10,
                                    nbdevup=2,
                                    nbdevdn=2)

    bars['bband_u_cl'] = upper_cl / mult
    bars['bband_m_cl'] = middle_cl / mult
    bars['bband_l_cl'] = lower_cl / mult
    bars['bband_u_cl_diff'] = bars['bband_u_cl'] - bars[c]
    bars['bband_m_cl_diff'] = bars['bband_m_cl'] - bars[c]
    bars['bband_l_cl_diff'] = bars['bband_l_cl'] - bars[c]
    bars['bband_u_cl_diff_hi'] = bars['bband_u_cl'] - bars[h]
    bars['bband_l_cl_diff_lo'] = bars['bband_l_cl'] - bars[l]
    # bars['bband_u_cl'].fillna(method='bfill', inplace=True)
    # bars['bband_m_cl'].fillna(method='bfill', inplace=True)
    # bars['bband_l_cl'].fillna(method='bfill', inplace=True)

    upper_tp, middle_tp, lower_tp = talib.BBANDS(mult_tp,
                                    timeperiod=10,
                                    nbdevup=2,
                                    nbdevdn=2)

    bars['bband_u_tp'] = upper_tp / mult
    bars['bband_m_tp'] = middle_tp / mult
    bars['bband_l_tp'] = lower_tp / mult
    bars['bband_u_tp_diff'] = bars['bband_u_tp'] - bars['typical_price']
    bars['bband_m_tp_diff'] = bars['bband_m_tp'] - bars['typical_price']
    bars['bband_l_tp_diff'] = bars['bband_l_tp'] - bars['typical_price']
    bars['bband_u_tp_diff_hi'] = bars['bband_u_tp'] - bars[h]
    bars['bband_l_tp_diff_lo'] = bars['bband_l_tp'] - bars[l]
    # think this is already taken care of at the end...check
    # bars['bband_u_tp'].fillna(method='bfill', inplace=True)
    # bars['bband_m_tp'].fillna(method='bfill', inplace=True)
    # bars['bband_l_tp'].fillna(method='bfill', inplace=True)

    # Double Exponential Moving Average
    bars['dema_cl'] = talib.DEMA(mult_close, timeperiod=30) / mult
    bars['dema_tp'] = talib.DEMA(mult_tp, timeperiod=30) / mult
    bars['dema_cl_diff'] = bars['dema_cl'] - bars[c]
    bars['dema_tp_diff'] = bars['dema_tp'] - bars['typical_price']


    # exponential moving Average
    bars['ema_cl'] = talib.EMA(mult_close, timeperiod=30) / mult
    bars['ema_tp'] = talib.EMA(mult_tp, timeperiod=30) / mult
    bars['ema_cl_diff'] = bars['ema_cl'] - bars[c]
    bars['ema_tp_diff'] = bars['ema_tp'] - bars['typical_price']

    # Hilbert Transform - Instantaneous Trendline - like a mva but a bit different, should probably take slope or
    # use in another indicator
    bars['ht_tl_cl'] = talib.HT_TRENDLINE(mult_close) / mult
    bars['ht_tl_tp'] = talib.HT_TRENDLINE(mult_tp) / mult
    bars['ht_tl_cl_diff'] = bars['ht_tl_cl'] - bars[c]
    bars['ht_tl_tp_diff'] = bars['ht_tl_tp'] - bars['typical_price']

    # KAMA - Kaufman's Adaptative Moving Average -- need to take slope or something
    bars['kama_cl'] = talib.KAMA(mult_close, timeperiod=30) / mult
    bars['kama_tp'] = talib.KAMA(mult_tp, timeperiod=30) / mult
    bars['kama_cl_diff'] = bars['kama_cl'] - bars[c]
    bars['kama_tp_diff'] = bars['kama_tp'] - bars['typical_price']

    # MESA Adaptive Moving Average -- getting TA_BAD_PARAM error
    # mama_cl, fama_cl = talib.MAMA(mult_close, fastlimit=100, slowlimit=100) / mult
    # mama_tp, fama_tp = talib.MAMA(mult_tp, fastlimit=100, slowlimit=100) / mult
    # mama_cl_osc = (mama_cl - fama_cl) / mama_cl
    # mama_tp_osc = (mama_tp - fama_tp) / mama_tp
    # bars['mama_cl'] = mama_cl
    # bars['mama_tp'] = mama_tp
    # bars['fama_cl'] = fama_cl
    # bars['fama_tp'] = fama_tp
    # bars['mama_cl_osc'] = mama_cl_osc
    # bars['mama_tp_osc'] = mama_tp_osc

    # Moving average with variable period
    bars['mavp_cl'] = talib.MAVP(mult_close, np.arange(mult_close.shape[0]).astype(np.float64), minperiod=2, maxperiod=30, matype=0) / mult
    bars['mavp_tp'] = talib.MAVP(mult_tp, np.arange(mult_tp.shape[0]).astype(np.float64), minperiod=2, maxperiod=30, matype=0) / mult
    bars['mavp_cl_diff'] = bars['mavp_cl'] - bars[c]
    bars['mavp_tp_diff'] = bars['mavp_tp'] - bars['typical_price']

    # midpoint over period
    bars['midp_cl'] = talib.MIDPOINT(mult_close, timeperiod=14) / mult
    bars['midp_tp'] = talib.MIDPOINT(mult_tp, timeperiod=14) / mult
    bars['midp_cl_diff'] = bars['midp_cl'] - bars[c]
    bars['midp_tp_diff'] = bars['midp_tp'] - bars['typical_price']

    # midpoint price over period
    bars['midpr'] = talib.MIDPRICE(mult_high, mult_low, timeperiod=14) / mult
    bars['midpr_diff'] = bars['midpr'] - bars['typical_price']

    # parabolic sar
    bars['sar'] = talib.SAR(mult_high, mult_low, acceleration=0.02, maximum=0.2) / mult
    bars['sar_diff'] = bars['sar'] - bars['typical_price']
    # need to make an oscillator for this

    # simple moving average
    # 10 day
    bars['sma_10_cl'] = talib.SMA(mult_close, timeperiod=10) / mult
    bars['sma_10_tp'] = talib.SMA(mult_tp, timeperiod=10) / mult
    # 20 day
    bars['sma_20_cl'] = talib.SMA(mult_close, timeperiod=20) / mult
    bars['sma_20_tp'] = talib.SMA(mult_tp, timeperiod=20) / mult
    # 30 day
    bars['sma_30_cl'] = talib.SMA(mult_close, timeperiod=30) / mult
    bars['sma_30_tp'] = talib.SMA(mult_tp, timeperiod=30) / mult
    # 40 day
    bars['sma_40_cl'] = talib.SMA(mult_close, timeperiod=40) / mult
    bars['sma_40_tp'] = talib.SMA(mult_tp, timeperiod=40) / mult

    # triple exponential moving average
    bars['tema_cl'] = talib.TEMA(mult_close, timeperiod=30) / mult
    bars['tema_tp'] = talib.TEMA(mult_tp, timeperiod=30) / mult
    bars['tema_cl_diff'] = bars['tema_cl'] - bars[c]
    bars['tema_tp_diff'] = bars['tema_tp'] - bars['typical_price']

    # triangular ma
    bars['trima_cl'] = talib.TRIMA(mult_close, timeperiod=30) / mult
    bars['trima_tp'] = talib.TRIMA(mult_tp, timeperiod=30) / mult
    bars['trima_cl_diff'] = bars['trima_cl'] - bars[c]
    bars['trima_tp_diff'] = bars['trima_tp'] - bars['typical_price']

    # weighted moving average
    bars['wma_cl'] = talib.WMA(mult_close, timeperiod=30) / mult
    bars['wma_tp'] = talib.WMA(mult_tp, timeperiod=30) / mult
    bars['wma_cl_diff'] = bars['wma_cl'] - bars[c]
    bars['wma_tp_diff'] = bars['wma_tp'] - bars['typical_price']

    #### momentum indicators  -- for now left out those with unstable periods

    # Average Directional Movement Index - 0 to 100 I think
    bars['adx'] = talib.ADX(mult_high, mult_low, mult_close, timeperiod=14)

    # Average Directional Movement Index Rating
    bars['adxr'] = talib.ADXR(mult_high, mult_low, mult_close, timeperiod=14)

    # Absolute Price Oscillator
    # values around -100 to +100
    bars['apo_cl'] = talib.APO(mult_close, fastperiod=12, slowperiod=26, matype=0)
    bars['apo_tp'] = talib.APO(mult_tp, fastperiod=12, slowperiod=26, matype=0)

    # Aroon and Aroon Oscillator 0-100, so don't need to renormalize
    arup, ardn = talib.AROON(mult_high, mult_low, timeperiod=14)
    bars['arup'] = arup
    bars['ardn'] = ardn

    # linearly related to aroon, just aroon up - aroon down
    bars['aroonosc'] = talib.AROONOSC(mult_high, mult_low, timeperiod=14)

    # balance of power - ratio of values so don't need to re-normalize
    bars['bop'] = talib.BOP(mult_open, mult_high, mult_low, mult_close)

    # Commodity Channel Index
    # around -100 to + 100
    bars['cci'] = talib.CCI(mult_high, mult_low, mult_close, timeperiod=14)

    # Chande Momentum Oscillator
    bars['cmo_cl'] = talib.CMO(mult_close, timeperiod=14)
    bars['cmo_tp'] = talib.CMO(mult_tp, timeperiod=14)

    # dx - Directional Movement Index
    bars['dx'] = talib.DX(mult_high, mult_low, mult_close, timeperiod=14)

    # Moving Average Convergence/Divergence
    # https://www.quantopian.com/posts/how-does-the-talib-compute-macd-why-the-value-is-different
    # macd diff btw fast and slow EMA
    macd_cl, macdsignal_cl, macdhist_cl = talib.MACD(mult_close, fastperiod=12, slowperiod=26, signalperiod=9)
    bars['macd_cl'] = macd_cl / mult
    bars['macdsignal_cl'] = macdsignal_cl / mult
    bars['macdhist_cl'] = macdhist_cl / mult

    macd_tp, macdsignal_tp, macdhist_tp = talib.MACD(mult_tp, fastperiod=12, slowperiod=26, signalperiod=9)
    bars['macd_tp'] = macd_tp / mult
    bars['macdsignal_tp'] = macdsignal_tp / mult
    bars['macdhist_tp'] = macdhist_tp / mult

    # mfi - Money Flow Index
    bars['mfi'] = talib.MFI(mult_high, mult_low, mult_close, bars[v].values, timeperiod=14)

    # minus di - Minus Directional Indicator
    bars['mdi'] = talib.MINUS_DI(mult_high, mult_low, mult_close, timeperiod=14)

    # Minus Directional Movement
    bars['mdm'] = talib.MINUS_DM(mult_high, mult_low, timeperiod=14)

    # note: too small of a timeperiod will result in junk data...I think.  or at least very discretized
    bars['mom_cl'] = talib.MOM(mult_close, timeperiod=14) / mult
    # bars['mom_cl'].fillna(method='bfill', inplace=True)
    bars['mom_tp'] = talib.MOM(mult_tp, timeperiod=14) / mult
    # bars['mom_tp'].fillna(method='bfill', inplace=True)

    # plus di - Plus Directional Indicator
    bars['pldi'] = talib.PLUS_DI(mult_high, mult_low, mult_close, timeperiod=14)

    # Plus Directional Movement
    bars['pldm'] = talib.PLUS_DM(mult_high, mult_low, timeperiod=14)

    # percentage price Oscillator
    bars['ppo_cl'] = talib.PPO(mult_close, fastperiod=12, slowperiod=26, matype=0)
    bars['ppo_tp'] = talib.PPO(mult_tp, fastperiod=12, slowperiod=26, matype=0)

    # rate of change
    bars['roc_cl'] = talib.ROC(mult_close, timeperiod=10)
    bars['roc_tp'] = talib.ROC(mult_tp, timeperiod=10)

    # rocp - Rate of change Percentage: (price-prevPrice)/prevPrice
    bars['rocp_cl'] = talib.ROCP(mult_close, timeperiod=10)
    bars['rocp_tp'] = talib.ROCP(mult_tp, timeperiod=10)

    # rocr - Rate of change ratio: (price/prevPrice)
    bars['rocr_cl'] = talib.ROCR(mult_close, timeperiod=10)
    bars['rocr_tp'] = talib.ROCR(mult_tp, timeperiod=10)

    # Rate of change ratio 100 scale: (price/prevPrice)*100
    bars['rocr_cl_100'] = talib.ROCR100(mult_close, timeperiod=10)
    bars['rocr_tp_100'] = talib.ROCR100(mult_tp, timeperiod=10)

    # Relative Strength Index
    bars['rsi_cl'] = talib.RSI(mult_close, timeperiod=14)
    bars['rsi_tp'] = talib.RSI(mult_tp, timeperiod=14)

    # stochastic oscillator - % of price diffs, so no need to rescale
    slowk, slowd = talib.STOCH(mult_high, mult_low, mult_close, fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
    fastk, fastd = talib.STOCHF(mult_high, mult_low, mult_close, fastk_period=5, fastd_period=3, fastd_matype=0)
    bars['slowk'] = slowk
    bars['slowd'] = slowd
    bars['fastk'] = fastk
    bars['fastd'] = fastd

    # Stochastic Relative Strength Index
    fastk_cl, fastd_cl = talib.STOCHRSI(mult_close, timeperiod=14, fastk_period=5, fastd_period=3, fastd_matype=0)
    fastk_tp, fastd_tp = talib.STOCHRSI(mult_tp, timeperiod=14, fastk_period=5, fastd_period=3, fastd_matype=0)
    bars['strsi_cl_k'] = fastk_cl
    bars['strsi_cl_d'] = fastd_cl
    bars['strsi_tp_k'] = fastk_tp
    bars['strsi_tp_d'] = fastd_tp

    # trix - 1-day Rate-Of-Change (ROC) of a Triple Smooth EMA
    bars['trix_cl'] = talib.TRIX(mult_close, timeperiod=30)
    bars['trix_tp'] = talib.TRIX(mult_tp, timeperiod=30)

    # ultimate Oscillator - between 0 and 100
    bars['ultosc'] = talib.ULTOSC(mult_high, mult_low, mult_close, timeperiod1=7, timeperiod2=14, timeperiod3=28)

    # williams % r  -- 0 to 100
    bars['willr'] = talib.WILLR(mult_high, mult_low, mult_close, timeperiod=14)


    ### volume indicators
    # Chaikin A/D Line
    bars['ad'] = talib.AD(mult_high, mult_low, mult_close, bars[v].values)

    # Chaikin A/D Oscillator
    bars['adosc'] = talib.ADOSC(mult_high, mult_low, mult_close, bars[v].values, fastperiod=3, slowperiod=10)

    # on balance volume
    bars['obv_cl'] = talib.OBV(mult_close, bars[v].values)
    bars['obv_tp'] = talib.OBV(mult_tp, bars[v].values)


    ### volatility indicators
    # average true range
    bars['atr'] = talib.ATR(mult_high, mult_low, mult_close, timeperiod=14)

    # Normalized Average True Range
    bars['natr'] = talib.NATR(mult_high, mult_low, mult_close, timeperiod=14)

    # true range
    bars['trange'] = talib.TRANGE(mult_high, mult_low, mult_close) / mult


    ### Cycle indicators
    # Hilbert Transform - Dominant Cycle Period
    bars['ht_dcp_cl'] = talib.HT_DCPERIOD(mult_close)
    bars['ht_dcp_tp'] = talib.HT_DCPERIOD(mult_tp)

    # Hilbert Transform - Dominant Cycle Phase
    bars['ht_dcph_cl'] = talib.HT_DCPHASE(mult_close)
    bars['ht_dcph_tp'] = talib.HT_DCPHASE(mult_tp)

    # Hilbert Transform - Phasor Components
    inphase_cl, quadrature_cl = talib.HT_PHASOR(mult_close)
    inphase_tp, quadrature_tp = talib.HT_PHASOR(mult_tp)
    bars['ht_ph_cl'] = inphase_cl
    bars['ht_ph_tp'] = inphase_tp
    bars['ht_q_cl'] = quadrature_cl
    bars['ht_q_tp'] = quadrature_tp

    # Hilbert Transform - SineWave
    sine_cl, leadsine_cl = talib.HT_SINE(mult_close)
    sine_tp, leadsine_tp = talib.HT_SINE(mult_tp)
    bars['ht_s_cl'] = sine_cl
    bars['ht_s_tp'] = sine_tp
    bars['ht_ls_cl'] = leadsine_cl
    bars['ht_ls_tp'] = leadsine_tp

    # Hilbert Transform - Trend vs Cycle Mode
    bars['ht_tr_cl'] = talib.HT_TRENDMODE(mult_close)
    bars['ht_tr_tp'] = talib.HT_TRENDMODE(mult_tp)


    bars.fillna(method='bfill', inplace=True)

    if return_df:
        return bars


def reject_outliers(sr, iq_range=0.5):
    pcnt = (1 - iq_range) / 2
    qlow, median, qhigh = sr.dropna().quantile([pcnt, 0.50, 1-pcnt])
    iqr = qhigh - qlow
    return sr[ (sr - median).abs() <= iqr]


def remove_outliers(df):
    """
    removes outliers for EDA
    """
    data = {}
    for c in df.columns:
        print(c)
        data[c] = reject_outliers(df[c])

    return data


if __name__ == "__main__":
    pass
