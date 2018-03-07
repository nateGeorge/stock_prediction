from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, LSTM, Flatten, Embedding, GlobalMaxPooling1D
from keras.regularizers import l2
from keras.layers.core import Reshape
from keras.layers.wrappers import TimeDistributed
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import MaxPooling1D
from keras.initializers import glorot_normal
from keras.layers.pooling import GlobalAveragePooling1D
from keras.optimizers import RMSprop
from keras.layers.normalization import BatchNormalization
from keras.callbacks import History
import numpy as np
from keras.layers.advanced_activations import LeakyReLU
from keras_tqdm import TQDMNotebookCallback
import plotly
plotly.offline.init_notebook_mode()
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.graph_objs as go
import math
import pandas as pd

# hyperparameters
EPOCHS = 200
BATCH = 100


def create_nn_data(train_fs, test_fs):
    # NOTE: to use keras's RNN LSTM module our input must be reshaped to [samples, stepsize, window size]
    # our stepsize is 1 because we increment the time by 1 for each sample
    # window size is 30 currently
    X_trains = {}
    X_tests = {}
    for s in train_fs.keys():
        X_trains[s] = np.asarray(np.reshape(train_fs[s], (train_fs[s].shape[0], 1, train_fs[s].shape[1])))
        X_tests[s] = np.asarray(np.reshape(test_fs[s], (test_fs[s].shape[0], 1, test_fs[s].shape[1])))

    return X_trains, X_tests


def create_nn_data_pcts(train, test):
    # NOTE: to use keras's RNN LSTM module our input must be reshaped to [samples, stepsize, window size]
    # our stepsize is 1 because we increment the time by 1 for each sample
    # window size is 30 currently
    X_train = np.asarray(np.reshape(train, (train.shape[0], 1, train.shape[1])))
    X_test = np.asarray(np.reshape(test, (test.shape[0], 1, test.shape[1])))

    return X_train, X_test


def create_nn_data4conv1d(train_fs, test_fs):
    # NOTE: to use keras's RNN LSTM module our input must be reshaped to [samples, stepsize, window size]
    # our stepsize is 1 because we increment the time by 1 for each sample
    # window size is 30 currently
    X_trains = {}
    X_tests = {}
    for s in train_fs.keys():
        X_trains[s] = np.asarray(np.reshape(train_fs[s], (train_fs[s].shape[0], train_fs[s].shape[1], 1)))
        X_tests[s] = np.asarray(np.reshape(test_fs[s], (test_fs[s].shape[0], test_fs[s].shape[1], 1)))

    return X_trains, X_tests


def create_model_1(X_train):
    """
    Found that this is overfitting because the test data (val) loss
    goes down and then way up.
    """
    model = Sequential()
    model.add(LSTM(256, input_shape=X_train.shape[1:], activation=None, return_sequences=True))
    model.add(elu)
    model.add(Dropout(0.5))
    model.add(LSTM(256, activation=None))
    model.add(LeakyReLU())
    model.add(Dropout(0.5))
    model.add(Dense(1))

    # build model using keras documentation recommended optimizer initialization
    optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

    # compile the model
    model.compile(loss='mean_squared_error', optimizer=optimizer)

    return model


def create_model_complex(X_train):
    """
    adding 2 more dense layers with dropout
    """
    model = Sequential()
    model.add(LSTM(256,
                    input_shape=X_train.shape[1:],
                    activation=None,
                    kernel_initializer='glorot_normal',
                    kernel_regularizer=l2(0.01),
                    bias_regularizer=l2(0.01),
                    return_sequences=True))
    model.add(LeakyReLU())
    # model.add(Dropout(0.5))
    model.add(LSTM(256,
                    activation=None,
                    kernel_initializer='glorot_normal',
                    kernel_regularizer=l2(0.01),
                    bias_regularizer=l2(0.01)))
    model.add(LeakyReLU())
    model.add(Dense(256, kernel_initializer='glorot_normal'))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(Dropout(0.5))
    model.add(Reshape((-1, 1)))
    model.add(Conv1D(64,
                    30,
                    strides=1,
                    kernel_initializer='glorot_normal',
                    padding='valid',
                    activation=None))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    # https://github.com/fchollet/keras/issues/4403 note on TimeDistributed
    model.add(MaxPooling1D(pool_size=2,
                            strides=2,
                            padding='valid'))
    model.add(Conv1D(128,
                    30,
                    strides=1,
                    kernel_initializer='glorot_normal',
                    padding='valid',
                    activation=None))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    # https://github.com/fchollet/keras/issues/4403 note on TimeDistributed
    model.add(MaxPooling1D(pool_size=2,
                            strides=2,
                            padding='valid'))
    model.add(Flatten())
    model.add(Dense(1024, kernel_initializer='glorot_normal'))
    model.add(BatchNormalization())
    model.add(LeakyReLU)
    model.add(Dropout(0.5))
    model.add(Dense(512, kernel_initializer='glorot_normal'))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(Dropout(0.5))
    model.add(Dense(128, kernel_initializer='glorot_normal'))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(Dense(1, kernel_initializer='glorot_normal'))

    # build model using keras documentation recommended optimizer initialization
    optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

    # compile the model
    model.compile(loss='mean_squared_error', optimizer=optimizer)

    return model


def create_model(X_train):
    """
    loss of 0.11 with 90 days history and 5 days prediction
    """
    model = Sequential()
    model.add(LSTM(256,
                    input_shape=X_train.shape[1:],
                    activation=None,
                    kernel_regularizer=l2(0.01),
                    bias_regularizer=l2(0.01),
                    return_sequences=True))
    model.add(LeakyReLU())
    # model.add(Dropout(0.5))
    model.add(LSTM(256,
                    activation=None,
                    kernel_regularizer=l2(0.01),
                    bias_regularizer=l2(0.01)))
    model.add(LeakyReLU())
    model.add(Dense(256))
    model.add(Dropout(0.5))
    model.add(Reshape((-1, 1)))
    model.add(Conv1D(64,
                    15,
                    strides=1,
                    padding='valid',
                    activation=None))
    # https://github.com/fchollet/keras/issues/4403 note on TimeDistributed
    model.add(MaxPooling1D(pool_size=2,
                            strides=2,
                            padding='valid'))
    model.add(Conv1D(128,
                    15,
                    strides=1,
                    padding='valid',
                    activation=None))
    # https://github.com/fchollet/keras/issues/4403 note on TimeDistributed
    model.add(MaxPooling1D(pool_size=2,
                            strides=2,
                            padding='valid'))
    model.add(Flatten())
    model.add(Dense(64))
    # model.add(Dropout(0.5))
    model.add(Dense(1))

    # build model using keras documentation recommended optimizer initialization
    optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

    # compile the model
    model.compile(loss='mean_squared_error', optimizer=optimizer)

    return model


def create_conv1d_model(X_train):
    """

    """
    model = Sequential()
    # example here: https://gist.github.com/jkleint/1d878d0401b28b281eb75016ed29f2ee
    model.add(Conv1D(64,
                    30,
                    strides=1,
                    padding='valid',
                    kernel_initializer='glorot_normal',
                    activation=None,
                    input_shape=(X_train.shape[1], 1)
                    ))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    # https://github.com/fchollet/keras/issues/4403 note on TimeDistributed
    model.add(MaxPooling1D(pool_size=2,
                            strides=2,
                            padding='valid'))
    model.add(Conv1D(256,
                    30,
                    strides=1,
                    padding='valid',
                    kernel_initializer='glorot_normal',
                    activation=None,
                    input_shape=(X_train.shape[1], 1)
                    ))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    # https://github.com/fchollet/keras/issues/4403 note on TimeDistributed
    model.add(MaxPooling1D(pool_size=2,
                            strides=2,
                            padding='valid'))
    # model.add(Flatten())  # dimensions were too big with this
    model.add(GlobalAveragePooling1D())
    model.add(Dense(256, kernel_initializer='glorot_normal'))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(Dropout(0.5))
    model.add(Dense(128, kernel_initializer='glorot_normal'))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(Dropout(0.5))
    model.add(Dense(1))

    # build model using keras documentation recommended optimizer initialization
    optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

    # compile the model
    model.compile(loss='mean_squared_error', optimizer=optimizer)

    return model


def create_model_lstm(X_train):
    """

    """
    model = Sequential()
    model.add(LSTM(256,
                    input_shape=X_train.shape[1:],
                    activation=None,
                    kernel_regularizer=l2(0.01),
                    bias_regularizer=l2(0.01),
                    return_sequences=True))
    model.add(LeakyReLU())
    # model.add(Dropout(0.5))
    model.add(LSTM(256,
                    activation=None,
                    kernel_regularizer=l2(0.01),
                    bias_regularizer=l2(0.01)))
    model.add(LeakyReLU())
    model.add(Dense(256))
    model.add(Dropout(0.5))
    model.add(Reshape((-1, 1)))
    model.add(Conv1D(64,
                    15,
                    strides=1,
                    padding='valid',
                    activation=None))
    # https://github.com/fchollet/keras/issues/4403 note on TimeDistributed
    model.add(MaxPooling1D(pool_size=2,
                            strides=2,
                            padding='valid'))
    model.add(Conv1D(128,
                    15,
                    strides=1,
                    padding='valid',
                    activation=None))
    # https://github.com/fchollet/keras/issues/4403 note on TimeDistributed
    model.add(MaxPooling1D(pool_size=2,
                            strides=2,
                            padding='valid'))
    model.add(Flatten())
    model.add(Dense(64))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(Dense(1))

    # compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    return model


def embed_model(X_train):
    model = Sequential()

    max_features = math.ceil(X_train.ravel().max())
    print('max_features for embed layer: ', max_features)
    embedding_dims = 50

    model.add(Embedding(max_features,
                        embedding_dims,
                        input_length=X_train.shape[1],
                        embeddings_regularizer=l2(1e-4)))
    model.add(Dropout(0.2))

    model.add(Conv1D(32, 3, padding='valid', activation='relu', strides=1))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(MaxPooling1D(pool_size=2,
                            strides=2,
                            padding='valid'))
    model.add(Conv1D(64, 3, padding='valid', activation='relu', strides=1))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(GlobalMaxPooling1D())

    model.add(Dense(100))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    print('Complete.')
    return model


def fit_model_nb(model, X_train, train_t, X_test, test_t):
    # for fitting the model in a jupyter notebook
    history = History()
    model.fit(X_train,
                train_t,
                epochs=EPOCHS,
                batch_size=BATCH,
                validation_data=[X_test, test_t],
                verbose=0,
                callbacks=[TQDMNotebookCallback(), history])

    return history


def fit_model(model, X_train, train_t, X_test, test_t):
    history = History()
    model.fit(X_train,
                train_t,
                epochs=EPOCHS,
                batch_size=BATCH,
                validation_data=[X_test, test_t],
                verbose=1,
                callbacks=[history])

    return history


def fit_model_silent(model, X_train, train_t, X_test, test_t, epochs=EPOCHS):
    history = History()
    model.fit(X_train,
                train_t,
                epochs=epochs,
                batch_size=BATCH,
                validation_data=[X_test, test_t],
                verbose=0,
                callbacks=[history])

    return history


def plot_losses(history):
    """
    Plots train and val losses from neural net training.
    """
    trace0 = go.Scatter(
        x = history.epoch,
        y = history.history['loss'],
        mode = 'lines+markers',
        name = 'loss'
    )
    trace1 = go.Scatter(
        x = history.epoch,
        y = history.history['val_loss'],
        mode = 'lines+markers',
        name = 'test loss'
    )
    f = iplot({'data':[trace0, trace1]})


def plot_data_preds_scaled(model, stock, scaled_ts, scaled_fs, dates, train_test='all', train_frac=0.85, future_days=5):
    if train_test == 'all':
        # vertical line should be on the first testing set point
        dates = dates[stock]
        train_size = int(train_frac * dates.shape[0])
        print(train_size)
        feats = scaled_fs[stock]
        print(feats.shape)
        for_pred = feats.reshape(feats.shape[0],
                                1, feats.shape[1])
        preds = model.predict(for_pred).ravel()
        print(max([max(scaled_ts[stock].ravel()), max(preds)]))
        layout = {'shapes': [
        {
            'type': 'rect',
            # stupid hack to deal with pandas issue
            'x0': dates[train_size].date().strftime('%Y-%m-%d'),
            'y0': 1.1 * min([min(scaled_ts[stock].ravel()), min(preds)]),
            'x1': dates[-1].date().strftime('%Y-%m-%d'),
            'y1': 1.1 * max([max(scaled_ts[stock].ravel()), max(preds)]),
            'line': {
                'color': 'rgb(255, 0, 0)',
                'width': 2,
            },
            'fillcolor': 'rgba(128, 0, 128, 0.05)',
        },
        {
            'type': 'line',
            # first line is just before first point of test set
            'x0': dates[train_size+future_days].date().strftime('%Y-%m-%d'),
            'y0': 1.1 * min([min(scaled_ts[stock].ravel()), min(preds)]),
            'x1': dates[train_size+future_days].date().strftime('%Y-%m-%d'),
            'y1': 1.1 * max([max(scaled_ts[stock].ravel()), max(preds)]),
            'line': {
                'color': 'rgb(0, 255, 0)',
                'width': 2,
            }
        }]}

        trace0 = go.Scatter(
            x = dates,
            y = scaled_ts[stock].ravel(),
            mode = 'lines+markers',
            name = 'actual'
        )
        trace1 = go.Scatter(
            x = dates,
            y = preds,
            mode = 'lines+markers',
            name = 'predictions'
        )
        f = iplot({'data':[trace0, trace1], 'layout':layout})
    elif train_test == 'train':
        train_size = int(train_frac * dfs[stock].shape[0])
        feats = scaled_fs[stock][:train_size]
        for_pred = feats.reshape(feats.shape[0],
                                1, feats.shape[1])
        trace0 = go.Scatter(
            x = dfs[stock].iloc[:train_size].index,
            y = scaled_ts[stock].ravel(),
            mode = 'lines+markers',
            name = 'actual'
        )
        trace1 = go.Scatter(
            x = dfs[stock].iloc[:train_size].index,
            y = model.predict(for_pred).ravel(),
            mode = 'lines+markers',
            name = 'predictions'
        )
        f = iplot([trace0, trace1])
    elif train_test == 'test':
        train_size = int(train_frac * dfs[stock].shape[0])
        feats = scaled_fs[stock][train_size:]
        for_pred = feats.reshape(feats.shape[0],
                                1, feats.shape[1])
        trace0 = go.Scatter(
            x = dfs[stock].iloc[train_size:].index,
            y = scaled_ts[stock].ravel(),
            mode = 'lines+markers',
            name = 'actual'
        )
        trace1 = go.Scatter(
            x = dfs[stock].iloc[train_size:].index,
            y = model.predict(for_pred).ravel(),
            mode = 'lines+markers',
            name = 'predictions'
        )
        f = iplot([trace0, trace1])
    else:
        print('error!  You have to supply train_test as \'all\', \'train\', or \'test\'')


def plot_data_preds_unscaled(model, stock, t_scalers, scaled_ts, scaled_fs, targs, dates, datapoints=300, train_frac=0.85, future_days=5):
    dates = dates[stock]
    train_size = int(train_frac * dates.shape[0])

    for_preds = scaled_fs[stock].reshape(scaled_fs[stock].shape[0],
                                        1, scaled_fs[stock].shape[1])
    preds = model.predict(for_preds).ravel()
    unscaled_preds = t_scalers[stock].reform_data(preds, orig=True)

    if datapoints == 'all':
        datapoints = dates.shape[0]

    layout = {'shapes': [
    {
        'type': 'rect',
        # first line is just before first point of test set
        'x0': dates[train_size].date().strftime('%Y-%m-%d'),
        'y0': 1.1 * min([min(targs[stock][-datapoints:]), min(unscaled_preds.ravel()[-datapoints:])]),
        'x1': dates[-1].date().strftime('%Y-%m-%d'),
        'y1': 1.1 * max([max(targs[stock][-datapoints:]), max(unscaled_preds.ravel()[-datapoints:])]),
        'line': {
            'color': 'rgb(255, 0, 0)',
            'width': 2,
        },
        'fillcolor': 'rgba(128, 0, 128, 0.05)',
    },
    {
        'type': 'line',
        # first line is just before first point of test set
        'x0': dates[train_size+future_days].date().strftime('%Y-%m-%d'),
        'y0': 1.1 * min([min(targs[stock][-datapoints:]), min(unscaled_preds.ravel()[-datapoints:])]),
        'x1': dates[train_size+future_days].date().strftime('%Y-%m-%d'),
        'y1': 1.1 * max([max(targs[stock][-datapoints:]), max(unscaled_preds.ravel()[-datapoints:])]),
        'line': {
            'color': 'rgb(0, 255, 0)',
            'width': 2,
        }
    }],
    'yaxis': {'title': 'GLD price'}}

    trace0 = go.Scatter(
        x = dates[-datapoints:],
        y = targs[stock][-datapoints:],
        mode = 'lines+markers',
        name = 'actual'
    )
    trace1 = go.Scatter(
        x = dates[-datapoints:],
        y = unscaled_preds.ravel()[-datapoints:],
        mode = 'lines+markers',
        name = 'predictions'
    )
    f = iplot({'data':[trace0, trace1], 'layout':layout})


def plot_data_preds_unscaled_future(model, stock, t_scalers, scaled_ts, scaled_fs, targs, dates, datapoints=300, future_days=20):
    """
    plots training data and future prices of unseen data
    """
    for_preds = scaled_fs[stock].reshape(scaled_fs[stock].shape[0],
                                        1, scaled_fs[stock].shape[1])
    preds = model.predict(for_preds).ravel()
    unscaled_preds = t_scalers[stock].reform_data(preds, orig=True)

    if datapoints == 'all':
        datapoints = dates[stock].shape[0]

    # need to generate more dates for the unseen data
    pred_dates = dfs[stock].index + pd.Timedelta(str(future_days) + ' days')

    trace0 = go.Scatter(
        x = pred_dates[-datapoints:],
        y = targs[stock][-datapoints:],
        mode = 'lines+markers',
        name = 'actual'
    )
    trace1 = go.Scatter(
        x = pred_dates[-datapoints:],
        y = unscaled_preds.ravel()[-datapoints:],
        mode = 'lines+markers',
        name = 'predictions'
    )
    f = iplot({'data':[trace0, trace1]})


def plot_data_preds_unscaled_embed(model, stock, dfs, t_scalers, scaled_ts, scaled_fs, targs, datapoints=300, train_frac=0.85):
    train_size = int(train_frac * dfs[stock].shape[0])

    for_preds = scaled_fs[stock]
    preds = model.predict(for_preds).ravel()
    unscaled_preds = t_scalers[stock].reform_data(preds, orig=True)

    if datapoints == 'all':
        datapoints = dfs[stock].shape[0]

    layout = {'shapes': [
    {
        'type': 'rect',
        # stupid hack to deal with pandas issue
        'x0': dfs[stock].iloc[train_size:train_size + 1].index[0].date().strftime('%Y-%m-%d'),
        'y0': 1.1 * min([min(targs[stock][-datapoints:]), min(unscaled_preds.ravel()[-datapoints:])]),
        'x1': dfs[stock].iloc[-2:-1].index[0].date().strftime('%Y-%m-%d'),
        'y1': 1.1 * max([max(targs[stock][-datapoints:]), max(unscaled_preds.ravel()[-datapoints:])]),
        'line': {
            'color': 'rgb(255, 0, 0)',
            'width': 2,
        },
        'fillcolor': 'rgba(128, 0, 128, 0.05)',
    }]}

    trace0 = go.Scatter(
        x = dfs[stock].index[-datapoints:],
        y = targs[stock][-datapoints:],
        mode = 'lines+markers',
        name = 'actual'
    )
    trace1 = go.Scatter(
        x = dfs[stock].index[-datapoints:],
        y = unscaled_preds.ravel()[-datapoints:],
        mode = 'lines+markers',
        name = 'predictions'
    )
    f = iplot({'data':[trace0, trace1], 'layout':layout})


def plot_data_preds_scaled_conv1d(model, stock, dfs, scaled_ts, scaled_fs, train_test='all', train_frac=0.85):
    if train_test == 'all':
        # vertical line should be on the first testing set point
        train_size = int(train_frac * dfs[stock].shape[0])
        print(train_size)
        feats = scaled_fs[stock]
        for_pred = feats.reshape(feats.shape[0],
                                feats.shape[1],
                                1)
        preds = model.predict(for_pred).ravel()
        print(max([max(scaled_ts[stock].ravel()), max(preds)]))
        layout = {'shapes': [
        {
            'type': 'rect',
            # stupid hack to deal with pandas issue
            'x0': dfs[stock].iloc[train_size:train_size + 1].index[0].date().strftime('%Y-%m-%d'),
            'y0': 1.1 * min([min(scaled_ts[stock].ravel()), min(preds)]),
            'x1': dfs[stock].iloc[-2:-1].index[0].date().strftime('%Y-%m-%d'),
            'y1': 1.1 * max([max(scaled_ts[stock].ravel()), max(preds)]),
            'line': {
                'color': 'rgb(255, 0, 0)',
                'width': 2,
            },
            'fillcolor': 'rgba(128, 0, 128, 0.05)',
        }]}
        trace0 = go.Scatter(
            x = dfs[stock].index,
            y = scaled_ts[stock].ravel(),
            mode = 'lines+markers',
            name = 'actual'
        )
        trace1 = go.Scatter(
            x = dfs[stock].index,
            y = preds,
            mode = 'lines+markers',
            name = 'predictions'
        )
        f = iplot({'data':[trace0, trace1], 'layout':layout})
    elif train_test == 'train':
        train_size = int(train_frac * dfs[stock].shape[0])
        feats = scaled_fs[stock][:train_size]
        for_pred = feats.reshape(feats.shape[0],
                                feats.shape[1],
                                1)
        trace0 = go.Scatter(
            x = dfs[stock].iloc[:train_size].index,
            y = scaled_ts[stock].ravel(),
            mode = 'lines+markers',
            name = 'actual'
        )
        trace1 = go.Scatter(
            x = dfs[stock].iloc[:train_size].index,
            y = model.predict(for_pred).ravel(),
            mode = 'lines+markers',
            name = 'predictions'
        )
        f = iplot([trace0, trace1])
    elif train_test == 'test':
        train_size = int(train_frac * dfs[stock].shape[0])
        feats = scaled_fs[stock][train_size:]
        for_pred = feats.reshape(feats.shape[0],
                                feats.shape[1],
                                1)
        trace0 = go.Scatter(
            x = dfs[stock].iloc[train_size:].index,
            y = scaled_ts[stock].ravel(),
            mode = 'lines+markers',
            name = 'actual'
        )
        trace1 = go.Scatter(
            x = dfs[stock].iloc[train_size:].index,
            y = model.predict(for_pred).ravel(),
            mode = 'lines+markers',
            name = 'predictions'
        )
        f = iplot([trace0, trace1])
    else:
        print('error!  You have to supply train_test as \'all\', \'train\', or \'test\'')


def plot_data_preds_unscaled_conv1d(model, stock, dfs, t_scalers, scaled_fs, targs):
    for_preds = scaled_fs[stock].reshape(scaled_fs[stock].shape[0],
                                        scaled_fs[stock].shape[1],
                                        1)
    preds = model.predict(for_preds).ravel()
    unscaled_preds = t_scalers[stock].reform_data(preds, orig=True)

    datapoints = 300
    trace0 = go.Scatter(
        x = dfs[stock].index[-datapoints:],
        y = targs[stock][-datapoints:],
        mode = 'lines+markers',
        name = 'actual'
    )
    trace1 = go.Scatter(
        x = dfs[stock].index[-datapoints:],
        y = unscaled_preds.ravel()[-datapoints:],
        mode = 'lines+markers',
        name = 'predictions'
    )
    f = iplot([trace0, trace1])
