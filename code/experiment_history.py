# the following are for 30 days history and 5 days in the future prediction
# unless otherwise noted

def create_model(X_train):
    """
    loss of 0.3 after 400 epochs, batch size 500
    about same loss with 100-200 epochs with batch size 100
    """
    leaky_relu = LeakyReLU()

    model = Sequential()
    model.add(LSTM(256,
                    input_shape=X_train.shape[1:],
                    activation=None,
                    kernel_regularizer=regularizers.l2(0.01),
                    bias_regularizer=regularizers.l2(0.01),
                    return_sequences=True))
    model.add(leaky_relu)
    # model.add(Dropout(0.5))
    model.add(LSTM(256,
                    activation=None,
                    kernel_regularizer=regularizers.l2(0.01),
                    bias_regularizer=regularizers.l2(0.01)))
    model.add(leaky_relu)
    # model.add(Dropout(0.5))
    model.add(Dense(1))

    # build model using keras documentation recommended optimizer initialization
    optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

    # compile the model
    model.compile(loss='mean_squared_error', optimizer=optimizer)

    return model


def create_model(X_train):
    """
    took 500s to run and had a huge loss: 884585338.71587539

    """
    leaky_relu = LeakyReLU()

    model = Sequential()
    # example here: https://gist.github.com/jkleint/1d878d0401b28b281eb75016ed29f2ee
    model.add(Conv1D(32,
                    15,
                    strides=1,
                    padding='valid',
                    activation=None,
                    input_shape=(X_train.shape[1], 1)
                    ))
    # model.add(BatchNormalization())
    model.add(Activation('relu'))
    # https://github.com/fchollet/keras/issues/4403 note on TimeDistributed
    model.add(MaxPooling1D(pool_size=2,
                            strides=2,
                            padding='valid'))
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(256,
                    activation=None,
                    kernel_regularizer=regularizers.l2(0.01),
                    bias_regularizer=regularizers.l2(0.01),
                    return_sequences=True))
    # model.add(BatchNormalization())
    model.add(leaky_relu)
    # model.add(Dropout(0.5))
    model.add(LSTM(256,
                    activation=None,
                    kernel_regularizer=regularizers.l2(0.01),
                    bias_regularizer=regularizers.l2(0.01)))
    # model.add(BatchNormalization())
    model.add(leaky_relu)
    # model.add(Dropout(0.5))
    model.add(Dense(1))

    # build model using keras documentation recommended optimizer initialization
    optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

    # compile the model
    model.compile(loss='mean_squared_error', optimizer=optimizer)

    return model


def create_model(X_train):
    """
    loss of 0.24 after 200 epochs, batch size 100
    really no improvement adding two more FC layers of 128 and 64
    """
    leaky_relu = LeakyReLU()

    model = Sequential()
    model.add(LSTM(256,
                    input_shape=X_train.shape[1:],
                    activation=None,
                    kernel_regularizer=regularizers.l2(0.01),
                    bias_regularizer=regularizers.l2(0.01),
                    return_sequences=True))
    model.add(leaky_relu)
    # model.add(Dropout(0.5))
    model.add(LSTM(256,
                    activation=None,
                    kernel_regularizer=regularizers.l2(0.01),
                    bias_regularizer=regularizers.l2(0.01)))
    model.add(leaky_relu)
    model.add(Dense(128))
    model.add(Dropout(0.5))
    # model.add(Reshape())
    # model.add(Conv1D(32,
    #                 15,
    #                 strides=1,
    #                 padding='valid',
    #                 activation=None))
    model.add(Activation('relu'))
    # https://github.com/fchollet/keras/issues/4403 note on TimeDistributed
    # model.add(MaxPooling1D(pool_size=2,
    #                         strides=2,
    #                         padding='valid'))
    # model.add(Dropout(0.5))
    model.add(Dense(1))

    # build model using keras documentation recommended optimizer initialization
    optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

    # compile the model
    model.compile(loss='mean_squared_error', optimizer=optimizer)

    return model


def create_model(X_train):
    """
    loss about 0.27, runtime 21s
    adding another 64-dim FC layer at the end improved loss to 0.20
    """
    leaky_relu = LeakyReLU()

    model = Sequential()
    model.add(LSTM(256,
                    input_shape=X_train.shape[1:],
                    activation=None,
                    kernel_regularizer=regularizers.l2(0.01),
                    bias_regularizer=regularizers.l2(0.01),
                    return_sequences=True))
    model.add(leaky_relu)
    # model.add(Dropout(0.5))
    model.add(LSTM(256,
                    activation=None,
                    kernel_regularizer=regularizers.l2(0.01),
                    bias_regularizer=regularizers.l2(0.01)))
    model.add(leaky_relu)
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
    model.add(Flatten())
    # model.add(Dropout(0.5))
    model.add(Dense(1))

    # build model using keras documentation recommended optimizer initialization
    optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

    # compile the model
    model.compile(loss='mean_squared_error', optimizer=optimizer)

    return model


def create_model(X_train):
    """
    loss of 0.11 with 90 days history and 5 days prediction
    """
    leaky_relu = LeakyReLU()

    model = Sequential()
    model.add(LSTM(256,
                    input_shape=X_train.shape[1:],
                    activation=None,
                    kernel_regularizer=regularizers.l2(0.01),
                    bias_regularizer=regularizers.l2(0.01),
                    return_sequences=True))
    model.add(leaky_relu)
    # model.add(Dropout(0.5))
    model.add(LSTM(256,
                    activation=None,
                    kernel_regularizer=regularizers.l2(0.01),
                    bias_regularizer=regularizers.l2(0.01)))
    model.add(leaky_relu)
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


def create_model(X_train):
    """
    decreasing conv kernel to size 5 increased loss to 0.12
    increasing to 60 decreased loss slightly, at 0.117
    """
    leaky_relu = LeakyReLU()

    model = Sequential()
    model.add(LSTM(256,
                    input_shape=X_train.shape[1:],
                    activation=None,
                    kernel_regularizer=regularizers.l2(0.01),
                    bias_regularizer=regularizers.l2(0.01atom
                    return_sequences=True))
    model.add(leaky_relu)
    # model.add(Dropout(0.5))
    model.add(LSTM(256,
                    activation=None,
                    kernel_regularizer=regularizers.l2(0.01),
                    bias_regularizer=regularizers.l2(0.01)))
    model.add(leaky_relu)
    model.add(Dense(256))
    model.add(Dropout(0.5))
    model.add(Reshape((-1, 1)))
    model.add(Conv1D(64,
                    5,
                    strides=1,
                    padding='valid',
                    activation=None))
    # https://github.com/fchollet/keras/issues/4403 note on TimeDistributed
    model.add(MaxPooling1D(pool_size=2,
                            strides=2,
                            padding='valid'))
    model.add(Conv1D(128,
                    5,
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


def create_model(X_train):
    """
    adding 2 more dense layers with dropout actually made loss worse, 0.18
    """
    leaky_relu = LeakyReLU()

    model = Sequential()
    model.add(LSTM(256,
                    input_shape=X_train.shape[1:],
                    activation=None,
                    kernel_regularizer=regularizers.l2(0.01),
                    bias_regularizer=regularizers.l2(0.01),
                    return_sequences=True))
    model.add(leaky_relu)
    # model.add(Dropout(0.5))
    model.add(LSTM(256,
                    activation=None,
                    kernel_regularizer=regularizers.l2(0.01),
                    bias_regularizer=regularizers.l2(0.01)))
    model.add(leaky_relu)
    model.add(Dense(256))
    model.add(Dropout(0.5))
    model.add(Reshape((-1, 1)))
    model.add(Conv1D(64,
                    60,
                    strides=1,
                    padding='valid',
                    activation=None))
    # https://github.com/fchollet/keras/issues/4403 note on TimeDistributed
    model.add(MaxPooling1D(pool_size=2,
                            strides=2,
                            padding='valid'))
    model.add(Conv1D(128,
                    60,
                    strides=1,
                    padding='valid',
                    activation=None))
    # https://github.com/fchollet/keras/issues/4403 note on TimeDistributed
    model.add(MaxPooling1D(pool_size=2,
                            strides=2,
                            padding='valid'))
    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Dropout(0.5))
    model.add(Dense(512))
    model.add(Dropout(0.5))
    model.add(Dense(128))
    model.add(Dense(1))

    # build model using keras documentation recommended optimizer initialization
    optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

    # compile the model
    model.compile(loss='mean_squared_error', optimizer=optimizer)

    return model

def create_model(X_train):
    """
    adding 2 more dense layers with dropout, even with glorot initializers
    and batchnorm still gets a loss of 0.14
    decreasing conv kernels to 30 decreased loss to 0.128
    """
    leaky_relu = LeakyReLU()

    model = Sequential()
    model.add(LSTM(256,
                    input_shape=X_train.shape[1:],
                    activation=None,
                    kernel_initializer='glorot_normal',
                    kernel_regularizer=regularizers.l2(0.01),
                    bias_regularizer=regularizers.l2(0.01),
                    return_sequences=True))
    model.add(leaky_relu)
    # model.add(Dropout(0.5))
    model.add(LSTM(256,
                    activation=None,
                    kernel_initializer='glorot_normal',
                    kernel_regularizer=regularizers.l2(0.01),
                    bias_regularizer=regularizers.l2(0.01)))
    model.add(leaky_relu)
    model.add(Dense(256, kernel_initializer='glorot_normal'))
    model.add(BatchNormalization())
    model.add(leaky_relu)
    model.add(Dropout(0.5))
    model.add(Reshape((-1, 1)))
    model.add(Conv1D(64,
                    60,
                    strides=1,
                    kernel_initializer='glorot_normal',
                    padding='valid',
                    activation=None))
    model.add(BatchNormalization())
    model.add(leaky_relu)
    # https://github.com/fchollet/keras/issues/4403 note on TimeDistributed
    model.add(MaxPooling1D(pool_size=2,
                            strides=2,
                            padding='valid'))
    model.add(Conv1D(128,
                    60,
                    strides=1,
                    kernel_initializer='glorot_normal',
                    padding='valid',
                    activation=None))
    model.add(BatchNormalization())
    model.add(leaky_relu)
    # https://github.com/fchollet/keras/issues/4403 note on TimeDistributed
    model.add(MaxPooling1D(pool_size=2,
                            strides=2,
                            padding='valid'))
    model.add(Flatten())
    model.add(Dense(1024, kernel_initializer='glorot_normal'))
    model.add(BatchNormalization())
    model.add(leaky_relu)
    model.add(Dropout(0.5))
    model.add(Dense(512, kernel_initializer='glorot_normal'))
    model.add(BatchNormalization())
    model.add(leaky_relu)
    model.add(Dropout(0.5))
    model.add(Dense(128, kernel_initializer='glorot_normal'))
    model.add(BatchNormalization())
    model.add(leaky_relu)
    model.add(Dense(1, kernel_initializer='glorot_normal'))

    # build model using keras documentation recommended optimizer initialization
    optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

    # compile the model
    model.compile(loss='mean_squared_error', optimizer=optimizer)

    return model


def create_conv1d_model(X_train):
    """
    90 days history, 5 days future
    pretty poor, 0.4 loss
    also does horribly on the test data
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
    model.add(leaky_relu)
    # https://github.com/fchollet/keras/issues/4403 note on TimeDistributed
    model.add(MaxPooling1D(pool_size=2,
                            strides=2,
                            padding='valid'))
    # model.add(Flatten())  # dimensions were too big with this
    model.add(GlobalAveragePooling1D())
    model.add(Dense(512, kernel_initializer='glorot_normal'))
    model.add(BatchNormalization())
    model.add(leaky_relu)
    model.add(Dropout(0.5))
    model.add(Dense(128, kernel_initializer='glorot_normal'))
    model.add(BatchNormalization())
    model.add(leaky_relu)
    model.add(Dropout(0.5))
    model.add(Dense(1))

    # build model using keras documentation recommended optimizer initialization
    optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

    # compile the model
    model.compile(loss='mean_squared_error', optimizer=optimizer)

    return model
