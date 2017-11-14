# core
import os
import sys
import glob
import gc

# custom
sys.path.append('code')
sys.path.append('code/poloniex')
import prep_for_nn as pfn
import save_keras as sk
from utils import get_home_dir

# installed
from keras.layers import Input, Dense, Dropout, BatchNormalization, LSTM, Conv1D, MaxPooling1D, Activation, Flatten, Concatenate, Reshape
from keras.models import Model, load_model
from keras import optimizers
from keras.callbacks import EarlyStopping
from keras.regularizers import l1_l2
import keras.backend as K
import keras.losses
from poloniex import Poloniex
import tensorflow as tf
from plotly.offline import plot
from plotly.graph_objs import Scatter, Figure, Layout, Candlestick
import numpy as np
import pandas as pd
import statsmodels.api as sm


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
