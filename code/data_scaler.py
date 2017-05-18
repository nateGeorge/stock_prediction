from sklearn.preprocessing import StandardScaler as SS
import pandas as pd
import numpy as np

class data_scaler(object):
    """
    A class for scaling and unscaling data.
    """
    def __init__(self, mva=30, data=None):
        self.mva = mva
        self.scaler = SS()
        self.orig_data = data
        self.datasets = {}
        self.mvas = {}
        self.datasetcount = 0
        if data is not None:
            self.datasets['orig'] = data

    def set_mva(mva=30):
        """
        Allows the moving average period to be set.  Must be an integer.
        """
        self.mva = mva


    def transform_data(self, data=None):
        """
        This is for scaling the original data for use in machine learning.

        Takes a numpy array as input, divides by a moving average with period mva (integer),
        and returns the scaled data as well as the scaler
        and moving average (needed for returning the data to its original form).
        """
        if data is None and self.data is None:
            print('error! you need to supply the data here or when instantiating the class')
            return None
        elif data is None and self.data is not None:
            data = self.data
            self.datasets['orig'] = data

        # take the moving average with period self.mva
        rolling_mean = pd.Series(data).rolling(window=self.mva).mean()
        # fill in missing values at the beginning
        rolling_mean.iloc[:self.mva - 1] = rolling_mean.iloc[self.mva - 1]
        self.mvas['orig'] = rolling_mean
        mva_scaled = data / rolling_mean
        # use sklearn scaler to fit and transform scaled values
        scaled = self.scaler.fit_transform(mva_scaled.values.reshape(-1, 1))
        return scaled.ravel()

    def transform_more_data(self, data, mva=20):
        """
        Takes a numpy array (data) a standardscaler, and moving average data set as input,
        divides by the moving average and uses the scaler to scale it,
        and returns the scaled data.
        """
        self.datasetcount += 1
        print('this dataset is number ' + str(self.datasetcount) + '. Keep track of this for transforming back.')
        self.datasets['set' + str(self.datasetcount)] = data
        rolling_mean = pd.Series(data).rolling(window=self.mva).mean()
        # fill in missing values at the beginning
        rolling_mean.iloc[:self.mva - 1] = rolling_mean.iloc[self.mva - 1]
        self.mvas['set' + str(self.datasetcount)] = rolling_mean
        mva_scaled = data / rolling_mean
        scaled = self.scaler.transform(mva_scaled.values.reshape(-1, 1))
        return scaled.ravel()


    def reform_data(self, data, datasetnum=None, orig=False):
        """
        Re-constructs original data from the transformed data.  Requires the dataset number or to specify
        that
        """
        if orig is True and datasetnum is not None:
            print('error! must only supply original or supplimentary dataset')

        unscaled = self.scaler.inverse_transform(data)
        if datasetnum is not None:
            mva = self.mvas['set' + str(datasetnum)]
        elif orig == True:
            mva = self.mvas['orig']
        unrolled = unscaled * mva
        return unrolled


def scale_datasets(feats, targs):
    """
    Takes two dicts with keys as stock tickers (strings) and values as numpy
    arrays, and returns dicts with scaled targets and features.
    """
    t_scalers = {}  # target scalers dict
    scaled_ts = {}  # scaled targets
    f_scalers = {}  # feature scalers dict - dict of lists with stock tickers
    # as keys
    scaled_fs = {}  # scaled features
    for s in feats.keys():
        # scale and save targets
        t_ds = data_scaler(mva=30)
        scaled_targets = t_ds.transform_data(data=targs[s])
        t_scalers[s] = t_ds
        scaled_ts[s] = scaled_targets
        # scale and save features
        scalers = []
        scaled_data = []
        for i in range(feats[s].shape[1]):
            ds = data_scaler()
            scalers.append(ds)
            scaled_data.append(ds.transform_data(data=feats[s][:, i]))

        scaled_data = np.array(scaled_data).T
        f_scalers[s] = scalers
        scaled_fs[s] = scaled_data

    return t_scalers, scaled_ts, f_scalers, scaled_fs

def train_test_split_datasets(t_scalers, scaled_t, f_scalers, scaled_f, train_frac=0.85):
    train_fs = {}
    test_fs = {}
    train_ts = {}
    test_ts = {}
    for s in t_scalers.keys():
        train_size = int(train_frac * scaled_f[s].shape[0])
        train_f = scaled_f[s][:train_size]
        test_f = scaled_f[s][train_size:]
        train_t = scaled_t[s][:train_size].reshape(-1, 1)
        test_t = scaled_t[s][train_size:].reshape(-1, 1)

        train_fs[s] = train_f
        test_fs[s] = test_f
        train_ts[s] = train_t
        test_ts[s] = test_t

    return train_fs, test_fs, train_ts, test_ts
