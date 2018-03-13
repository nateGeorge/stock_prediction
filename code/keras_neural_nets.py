# core
import os
import sys
import glob
import gc

# custom
import save_keras_models as sk
import data_processing as dp

# installed
from keras.layers import Input, Dense, Dropout, BatchNormalization, LSTM, Conv1D, MaxPooling1D, Activation, Flatten, Concatenate, Reshape
from keras.models import Model, load_model
from keras import optimizers
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.regularizers import l1_l2
import keras.backend as K
import keras.losses
from poloniex import Poloniex
import tensorflow as tf
from plotly.offline import plot
from plotly.graph_objs import Scatter, Scattergl, Figure, Layout, Candlestick
import numpy as np
import pandas as pd
import statsmodels.api as sm


thismodule = sys.modules[__name__]  # used for getattr
HOME_DIR = '/home/nate/github/stock_prediction/'
plot_dir = HOME_DIR + 'plots/'
model_dir = 'models/'  # could be used as a place to store models

def set_weights(model, old_model_file):
    old_model = sk.load_network(old_model_file)
    for i, l in enumerate(model.layers):
        l.set_weights(old_model.layers[i].get_weights())

    del old_model

    return model


def stock_loss_mae_log(y_true, y_pred):
    # was using 8 and 2, but seemed to bias models towards just predicting everything negative
    # don't want the penalties too big, or it will just fit everything to a small number
    alpha1 = 1.  # penalty for predicting positive but actual is negative
    alpha2 = 1.  # penalty for predicting negative but actual is positive
    loss = tf.where(K.less(y_true * y_pred, 0), \
                     tf.where(K.less(y_true, y_pred), \
                                alpha1 * K.log(K.abs(y_true - y_pred) + 1), \
                                alpha2 * K.log(K.abs(y_true - y_pred) + 1)), \
                     K.log(K.abs(y_true - y_pred) + 1))

    return K.mean(loss, axis=-1)


# this needs to be performed before loading a model using this loss function
keras.losses.stock_loss_mae_log = stock_loss_mae_log


def big_dense(train_feats):
    """
    creates big dense model

    need to reshape input data like:
        train_feats.reshape(train_feats.shape[0], -1)

    by default loads weights from latest trained model on BTC_STR I think
    """
    # restart keras session (clear weights)
    K.clear_session()
    tf.reset_default_graph()

    timesteps = train_feats.shape[1]
    input_dim = train_feats.shape[2]
    inputs = Input(shape=(timesteps*input_dim, ))
    x = Dense(3000, activation='elu')(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.6)(x)
    x = Dense(2000, activation='elu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.6)(x)
    x = Dense(1000, activation='elu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(500, activation='elu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(100, activation='elu')(x)
    x = BatchNormalization()(x)
    x = Dense(1, activation='linear')(x)

    mod = Model(inputs, x)
    mod.compile(optimizer='adam', loss=stock_loss_mae_log)

    return mod


def smaller_conv1(train_feats):
    """
    """
    # restart keras session (clear weights)
    K.clear_session()
    tf.reset_default_graph()

    timesteps = train_feats.shape[1]
    input_dim = train_feats.shape[2]
    inputs = Input(shape=(timesteps, input_dim))
    x = Conv1D(filters=16, kernel_size=5, strides=1, kernel_initializer='glorot_normal')(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling1D()(x) # 2 x 2 : kernel x strides
    x = Activation('elu')(x)

    x = Conv1D(filters=32, kernel_size=5, strides=1, kernel_initializer='glorot_normal')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D()(x) # 2 x 2 : kernel x strides
    x = Activation('elu')(x)

    x = Conv1D(filters=64, kernel_size=5, strides=1, kernel_initializer='glorot_normal')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D()(x) # 2 x 2 : kernel x strides
    x = Activation('elu')(x)

    x = Flatten()(x)
    x = Dense(3000, activation='elu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = Dense(2000, activation='elu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = Dense(1000, activation='elu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(500, activation='elu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(100, activation='elu')(x)
    x = BatchNormalization()(x)
    output = Dense(1, activation='linear')(x)

    mod = Model(inputs, output)
    mod.compile(optimizer='adam', loss=stock_loss_mae_log)

    return mod


def deeper_conv1(train_feats):
    """
    """
    # restart keras session (clear weights)
    K.clear_session()
    tf.reset_default_graph()

    timesteps = train_feats.shape[1]
    input_dim = train_feats.shape[2]
    inputs = Input(shape=(timesteps, input_dim))
    x = Conv1D(filters=32, kernel_size=5, strides=1, kernel_initializer='glorot_normal')(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling1D()(x) # 2 x 2 : kernel x strides
    x = Activation('elu')(x)

    x = Conv1D(filters=64, kernel_size=5, strides=1, kernel_initializer='glorot_normal')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D()(x) # 2 x 2 : kernel x strides
    x = Activation('elu')(x)

    x = Conv1D(filters=128, kernel_size=5, strides=1, kernel_initializer='glorot_normal')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D()(x) # 2 x 2 : kernel x strides
    x = Activation('elu')(x)

    x = Flatten()(x)
    x = Dense(5000, activation='elu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = Dense(3000, activation='elu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = Dense(2000, activation='elu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = Dense(1000, activation='elu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(500, activation='elu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(100, activation='elu')(x)
    x = BatchNormalization()(x)
    output = Dense(1, activation='linear')(x)

    mod = Model(inputs, output)
    mod.compile(optimizer='adam', loss=stock_loss_mae_log)

    return mod


def big_conv1(train_feats):
    """
    creates big convolutional model
    requires more history
    """
    # restart keras session (clear weights)
    K.clear_session()
    tf.reset_default_graph()

    timesteps = train_feats.shape[1]
    input_dim = train_feats.shape[2]
    inputs = Input(shape=(timesteps, input_dim))
    x = Conv1D(filters=16, kernel_size=5, strides=1, kernel_initializer='glorot_normal')(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling1D()(x) # 2 x 2 : kernel x strides
    x = Activation('elu')(x)

    x = Conv1D(filters=32, kernel_size=5, strides=1, kernel_initializer='glorot_normal')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D()(x) # 2 x 2 : kernel x strides
    x = Activation('elu')(x)

    x = Conv1D(filters=64, kernel_size=5, strides=1, kernel_initializer='glorot_normal')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D()(x) # 2 x 2 : kernel x strides
    x = Activation('elu')(x)

    x = Conv1D(filters=128, kernel_size=5, strides=1, kernel_initializer='glorot_normal')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D()(x) # 2 x 2 : kernel x strides
    x = Activation('elu')(x)

    x = Flatten()(x)
    x = Dense(3000, activation='elu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.6)(x)
    x = Dense(2000, activation='elu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.6)(x)
    x = Dense(1000, activation='elu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(500, activation='elu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(100, activation='elu')(x)
    x = BatchNormalization()(x)
    output = Dense(1, activation='linear')(x)

    mod = Model(inputs, output)
    mod.compile(optimizer='adam', loss=stock_loss_mae_log)

    return mod


def get_latest_model(base, folder=None):
    files = glob.glob(model_dir + base + '*')
    if folder is not None:
        files = files + glob.glob(model_dir + folder + '/' + base + '*')

    if len(files) != 0:
        files.sort(key=os.path.getmtime)
        return files[-1]

    return None


def get_model_path(base, folder):
    if folder is None:
        model_file = model_dir + base + '.h5'
    else:
        model_file = model_dir + folder + '/' + base + '.h5'
        if not os.path.exists(model_dir + folder):
            os.mkdir(model_dir + folder)

    return model_file


def get_model_memory_usage(batch_size, model):
    import numpy as np
    from keras import backend as K

    shapes_mem_count = 0
    for l in model.layers:
        single_layer_mem = 1
        for s in l.output_shape:
            if s is None:
                continue
            single_layer_mem *= s
        shapes_mem_count += single_layer_mem

    trainable_count = np.sum([K.count_params(p) for p in set(model.trainable_weights)])
    non_trainable_count = np.sum([K.count_params(p) for p in set(model.non_trainable_weights)])

    total_memory = 4.0*batch_size*(shapes_mem_count + trainable_count + non_trainable_count)
    gbytes = np.round(total_memory / (1024.0 ** 3), 3)
    return gbytes


def generator_one_of_10(model, batch_size):
    """
    loads one of the 10 files of historical data for training
    """
    while True:
        # load data
        # for now, pick random file of 10 to load from
        i = np.random.choice(10)
        tr_feats, tr_targs, te_feats, te_targs, tr_indices, te_indices, stocks = dp.load_nn_data_one_set(i=i)
        # flatten data if dense model
        if model in ['big_dense']:
            train = tr_feats.reshape(tr_feats.shape[0], -1)
            test = te_feats.reshape(te_feats.shape[0], -1)
        elif model in ['big_conv1',
                        'conv1d_lstm',
                        'big_lstm_conv',
                        'smaller_conv1',
                        'conv1d_lstm_small',
                        'deeper_conv1']:
            train = tr_feats
            test = te_feats

        # shuffle data
        train_size = tr_feats.shape[0]
        idxs = np.arange(train_size)
        np.random.shuffle(idxs)
        train = train[idxs]
        tr_targs = tr_targs[idxs]

        # feed data in batches
        data_left = True
        start = 0
        end = batch_size
        while data_left:
            if end > train_size:
                end = train_size
                data_left = False

            tr_batch = train[start:end]
            tr_t_batch = tr_targs[start:end]
            start += batch_size
            end += batch_size
            yield tr_batch, tr_t_batch


def train_net(tr_feats=None,
                tr_targs=None,
                te_feats=None,
                te_targs=None,
                model='big_dense',
                random_init=False,
                latest_bias=False,
                folder=None,
                val_frac=0.15,
                batch_size=2000,
                generator=False):

    if not generator:
        if latest_bias:
            val_size = int(val_frac * train_targs.shape[0])
            train_eval = np.copy(xform_train)  # save original for evaluation
            train_eval_targs = np.copy(train_targs)
            start = -(test_size + val_size)
            for i in range(bias_factor):
                xform_train = np.vstack((xform_train, xform_train[start:-val_size, :, :]))
                train_targs = np.hstack((train_targs, train_targs[start:-val_size]))

    if random_init:
        latest_mod = None
    else:
        latest_mod = get_latest_model(base=model, folder=folder)

    if latest_mod is None:
        mod = getattr(thismodule, model)(tr_feats)
    else:
        mod = getattr(thismodule, model)(tr_feats)
        mod = set_weights(mod, latest_mod)

    if model in ['big_dense']:
        train = tr_feats.reshape(tr_feats.shape[0], -1)
        if latest_bias:
            train_eval = train_eval.reshape(train_eval.shape[0], -1)
        test = te_feats.reshape(te_feats.shape[0], -1)
    elif model in ['big_conv1',
                    'conv1d_lstm',
                    'big_lstm_conv',
                    'smaller_conv1',
                    'conv1d_lstm_small',
                    'deeper_conv1']:
        train = tr_feats
        test = te_feats

    es = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto')
    model_file = get_model_path(base=model, folder=folder)
    mc = ModelCheckpoint(model_file, save_best_only=True)
    tb = TensorBoard(log_dir='')
    cb = [es, mc]

    mem_use = get_model_memory_usage(batch_size, mod)
    print('expected memory usage:', mem_use)

    if generator:
        history = mod.fit_generator(generator_one_of_10(model=model, batch_size=batch_size),
                                    steps_per_epoch=1250,  # about 250K samples x10 for the future=10 hist=40 dataset, so about 2.5M total.  divide by 2k to get
                                    epochs=200)
    else:
        history = mod.fit(train,
                    tr_targs,
                    epochs=200,
                    validation_split=val_frac,
                    callbacks=cb,
                    batch_size=batch_size)

    print('saving as', model_file)
    sk.save_network(mod, model_file)

    best, lin_preds = get_best_thresh_linear_model(mod, test, te_targs)

    if latest_bias:
        plot_results(mod, train_eval, train_eval_targs, te_feats, te_targs, folder=folder, test_lin_preds=lin_preds)
        best_99pct = get_best_thresh(mod, train_eval, train_eval_targs, cln=False)
    else:
        plot_results(mod, tr_feats, tr_targs, te_feats, te_targs, folder=folder, test_lin_preds=lin_preds)
        best_99pct = get_best_thresh(mod, tr_feats, tr_targs, te_feats, te_targs, cln=False)


def reset_collect():
    """
    clears gpu memory
    """
    tf.reset_default_graph()
    K.clear_session()
    gc.collect()


def plot_results(mod, tr_feats, tr_targs, te_feats=None, te_targs=None, folder=None, test_lin_preds=None):
    # TODO: subplot with actual and predicted returns
    train_preds = mod.predict(tr_feats)[:, 0]
    train_score = mod.evaluate(tr_feats, tr_targs)
    # couldn't figure this out yet
    # train_score = K.run(stock_loss_mae_log(train_targs, train_preds))
    title = 'train preds vs actual (score = ' + str(train_score) + ')'
    if te_feats is None:
        title = 'full train preds vs actual (score = ' + str(train_score) + ')'
    data = [Scattergl(x=train_preds,
                    y=tr_targs,
                    mode='markers',
                    name='preds vs actual',
                    marker=dict(color=list(range(tr_targs.shape[0])),
                                colorscale='Portland',
                                showscale=True,
                                opacity=0.5)
                    )]
    layout = Layout(
        title=title,
        xaxis=dict(
            title='predictions'
        ),
        yaxis=dict(
            title='actual'
        )
    )
    fig = Figure(data=data, layout=layout)
    if folder is None:
        if te_feats is None:
            filename = plot_dir + 'full_train_preds_vs_actual.html'
        else:
            filename = plot_dir + 'train_preds_vs_actual.html'
    else:
        if not os.path.exists(plot_dir + folder):
            os.mkdir(plot_dir + folder)

        if te_feats is None:
            filename = plot_dir + folder + '/' + 'full_train_preds_vs_actual.html'
        else:
            filename = plot_dir + folder + '/' + 'train_preds_vs_actual.html'

    plot(fig, filename=filename, auto_open=False, show_link=False)

    del train_score

    if te_feats is not None:
        test_preds = mod.predict(te_feats)[:, 0]
        test_score = mod.evaluate(te_feats, te_targs)
        # test_score = K.run(stock_loss_mae_log(train_targs, train_preds))
        data = [Scattergl(x=test_preds,
                        y=te_targs,
                        mode='markers',
                        name='preds vs actual',
                        marker=dict(color=list(range(te_targs.shape[0])),
                                    colorscale='Portland',
                                    showscale=True,
                                    opacity=0.5)
                        )]
        if test_lin_preds is not None:
            line = Scatter(x=test_preds,
                            y=test_lin_preds,
                            mode='lines',
                            name='linear fit')
            data = data + [line]

        layout = Layout(
            title='test preds vs actual (score = ' + str(test_score) + ')',
            xaxis=dict(
                title='predictions'
            ),
            yaxis=dict(
                title='actual'
            )
        )
        fig = Figure(data=data, layout=layout)
        if folder is None:
            filename = plot_dir  + 'test_preds_vs_actual.html'
        else:
            filename = plot_dir + folder + '/' + 'test_preds_vs_actual.html'

        plot(fig, filename=filename, auto_open=False, show_link=False)

        del test_score


def get_best_thresh(mod, tr_feats, tr_targs, te_feats=None, te_targs=None, cln=True, verbose=False):
    # want to use both train and test sets, but want to plot each separately
    if te_feats is not None:
        train = np.vstack((tr_feats, te_feats))
        train_targs = np.hstack((tr_targs, te_targs))
        xform_test, test_targs = None, None

    train_preds = mod.predict(tr_feats)[:, 0]

    if cln:
        del mod
        reset_collect()

    # bin into 0.5% increments for the predictions
    max_pred = max(train_preds)
    best = max_pred
    hi = round_to_005(max_pred) # rounds to nearest half pct
    lo = round_to_005(hi - 0.005)  # 0.5% increments
    cumulative_agree = 0
    cumulative_pts = 0
    while lo > 0:
        mask = (train_preds >= lo) & (train_preds < hi)
        lo -= 0.005
        hi -= 0.005
        lo = round_to_005(lo)
        hi = round_to_005(hi)
        filtered = tr_targs[mask]
        if filtered.shape[0] == 0:
            continue

        pos = filtered[filtered >= 0].shape[0]
        cumulative_agree += pos
        cumulative_pts += filtered.shape[0]
        pct_agree = pos/filtered.shape[0]
        if pct_agree >= 0.99 and cumulative_agree >= 0.99:
            best = lo

        if verbose:
            print('interval:', '%.3f'%lo, '%.3f'%hi, ', pct agreement:', '%.3f'%pct_agree)

    return best


def get_best_thresh_linear_model(mod, te_feats, test_targs):
    preds = mod.predict(te_feats)[:, 0]
    X = sm.add_constant(preds, has_constant='add')
    y = test_targs
    lin_model = sm.OLS(y, X).fit()
    x_line = np.arange(min(preds), max(preds), 100)
    intercept, slope = lin_model.params  # intercept, slope
    # to get the value of x at y=0 (the x-intercept, where on average, the actual returns are positive)
    # take -b/m, or -intercetp/slope
    xint = -intercept / slope
    # add 10% for safety margin
    best = 1.1 * xint

    return best, lin_model.fittedvalues


def round_to_005(x):
    """
    rounds to nearest 0.005 and makes it pretty (avoids floating point 0.000001 nonsense)
    """
    res = round(x * 200) / 200
    return float('%.3f'%res)


def create_data(future=30, hist_points=40):
    """
    uses about 6 business weeks for future as default -- probably too big
    """
    import short_squeeze_eda as sse
    short_stocks = sse.get_stocks()
    dfs, sh_int, fin_sh = dp.load_stocks(stocks=short_stocks, verbose=True)
    dp.make_all_sh_future(sh_int, future=future, hist_points=hist_points, verbose=False)
    # del dfs
    # del fin_sh
    gc.collect()

    dp.make_nn_data(sh_int, hist_points=hist_points, future=future, make_fresh=True)


if __name__ == "__main__":
    # create_data()
    tr_feats, tr_targs, te_feats, te_targs, tr_indices, te_indices, stocks = dp.load_nn_data_one_set(i=9)
    # train_net(tr_feats, tr_targs, te_feats, te_targs)

    # used to finish up train_net when wasn't done yet
    # latest_mod = get_latest_model(base='big_dense', folder=None)
    # mod = getattr(thismodule, 'big_dense')(tr_feats)
    # mod = set_weights(mod, latest_mod)
    # best, lin_preds = get_best_thresh_linear_model(mod, te_feats.reshape(te_feats.shape[0], -1), te_targs)
    #
    # plot_results(mod, tr_feats.reshape(tr_feats.shape[0], -1), tr_targs, te_feats.reshape(te_feats.shape[0], -1), te_targs, folder=None, test_lin_preds=lin_preds)
    # best_99pct = get_best_thresh(mod, train=tr_feats.shape[0], -1), train_targs=train_targs, te_feats=te_feats.reshape(te_feats.shape[0], -1), test_targs=test_targs, cln=False)


    #train_net(tr_feats, tr_targs, te_feats, te_targs, model='deeper_conv1', generator=True)

    #dp.make_nn_data(sh_int, hist_points=hist_points, future=5, make_fresh=True)

    # train_net(tr_feats, tr_targs, te_feats, te_targs, model='big_conv1')  # wont work with 40 history points
