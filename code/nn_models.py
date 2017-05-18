from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import LSTM
import keras
from keras.callbacks import History
import numpy as np
from keras.layers.advanced_activations import LeakyReLU
from keras_tqdm import TQDMNotebookCallback
import plotly
plotly.offline.init_notebook_mode()
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.graph_objs as go


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


def create_model(X_train):
    leaky_relu = LeakyReLU()

    model = Sequential()
    model.add(LSTM(256, input_shape=X_train.shape[1:], activation=None, return_sequences=True))
    model.add(leaky_relu)
    model.add(Dropout(0.5))
    model.add(LSTM(256, activation=None))
    model.add(leaky_relu)
    model.add(Dropout(0.5))
    model.add(Dense(1))

    # build model using keras documentation recommended optimizer initialization
    optimizer = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

    # compile the model
    model.compile(loss='mean_squared_error', optimizer=optimizer)

    return model


def fit_model_nb(model, X_train, train_t):
    # for fitting the model in a jupyter notebook
    history = History()
    model.fit(X_train,
                train_t,
                epochs=1000,
                batch_size=500,
                verbose=0,
                callbacks=[TQDMNotebookCallback(), history])

    return history

def fit_model(model, X_train, train_t):
    history = History()
    model.fit(X_train,
                train_t,
                epochs=1000,
                batch_size=500,
                verbose=1,
                callbacks=[history])


def fit_model_silent(model, X_train, train_t):
    history = History()
    model.fit(X_train,
                train_t,
                epochs=10,
                batch_size=500,
                verbose=0,
                callbacks=[history])


def plot_data_preds_scaled(model, stock, dfs, scaled_ts, scaled_fs, train_test='all', train_frac=0.85):
    if train_test == 'all':
        # vertical line should be on the first testing set point
        train_size = int(train_frac * dfs[stock].shape[0])
        print(train_size)
        feats = scaled_fs[stock]
        for_pred = feats.reshape(feats.shape[0],
                                1, feats.shape[1])
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
            'fillcolor': 'rgba(128, 0, 128, 0.3)',
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






def plot_data_preds_unscaled(model, stock, dfs, t_scalers, scaled_fs, targs):
    for_preds = scaled_fs[stock].reshape(scaled_fs[stock].shape[0],
                                        1, scaled_fs[stock].shape[1])
    preds = model.predict(for_preds).ravel()
    unscaled_preds = t_scalers[stock].reform_data(preds, orig=True)

    trace0 = go.Scatter(
        x = dfs[stock].index,
        y = targs[stock],
        mode = 'lines+markers',
        name = 'actual'
    )
    trace1 = go.Scatter(
        x = dfs[stock].index,
        y = unscaled_preds.ravel(),
        mode = 'lines+markers',
        name = 'predictions'
    )
    f = iplot([trace0, trace1])
