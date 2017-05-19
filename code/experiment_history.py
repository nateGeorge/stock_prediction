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
