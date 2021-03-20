from keras.models import Model
from keras import backend as K
from keras.layers import Input, Flatten, TimeDistributed, Conv3D, MaxPooling3D, GRUCell, RNN
from keras.optimizers import Adam,SGD
from keras.regularizers import l2
from keras.layers.normalization import BatchNormalization
from model.layer import *
from model.loss import std_mae, std_rmse
from keras.layers import GlobalAveragePooling3D, GlobalMaxPooling3D, Reshape, Dense, multiply, Permute, Concatenate, Add, Activation, Lambda, Subtract

def attention_layer_fp(fea, ratio=8):
    fea = channel_att(fea, ratio)
    fea = x_att(fea, 2)
    fea = y_att(fea, 2)
    fea = z_att(fea, 2)
    return fea

def x_att(input_feature, ratio=8):
    print(K.image_data_format())
    axis_x = input_feature._keras_shape[1]

    shared_layer_one = Dense(axis_x // ratio, activation='relu', kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros')
    shared_layer_one_ = Dense(axis_x // ratio, activation='relu', kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros')
    shared_layer_two = Dense(axis_x, kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros')

    grid_feature = GlobalPoolingLayer_fp([2,3,4], axis_x)(input_feature)
    avg_pool = shared_layer_one(grid_feature)
    avg_pool = shared_layer_two(avg_pool)
    avg_pool = Reshape((axis_x, 1, 1, 1))(avg_pool)
    assert avg_pool._keras_shape[1:] == (axis_x, 1, 1, 1)

    max_f = Lambda(lambda x: tf.reduce_max(x, [2, 3, 4]))(input_feature)
    max_pool = shared_layer_one_(max_f)
    max_pool = shared_layer_two(max_pool)
    max_pool = Reshape((axis_x, 1, 1, 1))(max_pool)
    assert max_pool._keras_shape[1:] == (axis_x, 1, 1, 1)

    grid_feature = Add()([avg_pool, max_pool])
    grid_feature = Activation('sigmoid')(grid_feature)

    if K.image_data_format() == "channels_first":
        grid_feature = Permute((3, 1, 2))(grid_feature)

    return multiply([input_feature, grid_feature])

def y_att(input_feature, ratio=8):
    print(K.image_data_format())
    axis_y = input_feature._keras_shape[2]

    shared_layer_one = Dense(axis_y // ratio, activation='relu', kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros')
    shared_layer_one_ = Dense(axis_y // ratio, activation='relu', kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros')
    shared_layer_two = Dense(axis_y, kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros')

    grid_feature = GlobalPoolingLayer_fp([1, 3, 4], axis_y)(input_feature)
    avg_pool = shared_layer_one(grid_feature)
    avg_pool = shared_layer_two(avg_pool)
    avg_pool = Reshape((1, axis_y, 1, 1))(avg_pool)
    assert avg_pool._keras_shape[1:] == (1, axis_y, 1, 1)

    # max_f = tf.reduce_max(input_feature, [1, 3, 4])
    max_f = Lambda(lambda x: tf.reduce_max(x, [1, 3, 4]))(input_feature)
    max_pool = shared_layer_one_(max_f)
    max_pool = shared_layer_two(max_pool)
    max_pool = Reshape((1, axis_y, 1, 1))(max_pool)
    assert max_pool._keras_shape[1:] == (1, axis_y, 1, 1)

    grid_feature = Add()([avg_pool, max_pool])
    grid_feature = Activation('sigmoid')(grid_feature)

    if K.image_data_format() == "channels_first":
        grid_feature = Permute((3, 1, 2))(grid_feature)

    return multiply([input_feature, grid_feature])

def z_att(input_feature, ratio=8):
    print(K.image_data_format())
    axis_z = input_feature._keras_shape[3]

    shared_layer_one = Dense(axis_z // ratio, activation='relu', kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros')
    shared_layer_one_ = Dense(axis_z // ratio, activation='relu', kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros')
    shared_layer_two = Dense(axis_z, kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros')

    grid_feature = GlobalPoolingLayer_fp([1, 2, 4], axis_z)(input_feature)
    avg_pool = shared_layer_one(grid_feature)
    avg_pool = shared_layer_two(avg_pool)
    avg_pool = Reshape((1, 1, axis_z, 1))(avg_pool)
    assert avg_pool._keras_shape[1:] == (1, 1, axis_z, 1)

    max_f = Lambda(lambda x: tf.reduce_max(x, [1, 2, 4]))(input_feature)
    max_pool = shared_layer_one_(max_f)
    max_pool = shared_layer_two(max_pool)
    max_pool = Reshape((1, 1, axis_z, 1))(max_pool)
    assert max_pool._keras_shape[1:] == (1, 1, axis_z, 1)

    grid_feature = Add()([avg_pool, max_pool])
    grid_feature = Activation('sigmoid')(grid_feature)

    if K.image_data_format() == "channels_first":
        grid_feature = Permute((3, 1, 2))(grid_feature)

    return multiply([input_feature, grid_feature])



def attention_module_layer(fea, ratio=8):
    s_fea = spatial_att(fea)
    c_fea = channel_att(fea, ratio)

    coeff = Lambda(lambda x: tf.nn.sigmoid(x[0] + x[1]))([s_fea, c_fea])
    fea = Lambda(lambda x: tf.multiply(x[0], coeff) + tf.multiply(x[1], 1.0 - coeff))([s_fea, c_fea])
    return fea

def channel_att(input_feature, ratio=8):
    print(K.image_data_format())
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    channel = input_feature._keras_shape[channel_axis]

    shared_layer_one = Dense(channel // ratio,
                             activation='relu',
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros')
    shared_layer_two = Dense(channel,
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros')

    avg_pool = GlobalAveragePooling3D()(input_feature)
    avg_pool = Reshape((1, 1, 1, channel))(avg_pool)
    assert avg_pool._keras_shape[1:] == (1, 1, 1, channel)
    avg_pool = shared_layer_one(avg_pool)
    assert avg_pool._keras_shape[1:] == (1, 1, 1, channel // ratio)
    avg_pool = shared_layer_two(avg_pool)
    assert avg_pool._keras_shape[1:] == (1, 1, 1, channel)

    max_pool = GlobalMaxPooling3D()(input_feature)
    max_pool = Reshape((1, 1, 1, channel))(max_pool)
    assert max_pool._keras_shape[1:] == (1, 1, 1, channel)
    max_pool = shared_layer_one(max_pool)
    assert max_pool._keras_shape[1:] == (1, 1, 1, channel // ratio)
    max_pool = shared_layer_two(max_pool)
    assert max_pool._keras_shape[1:] == (1, 1, 1, channel)

    cbam_feature = Add()([avg_pool, max_pool])
    cbam_feature = Activation('sigmoid')(cbam_feature)

    if K.image_data_format() == "channels_first":
        cbam_feature = Permute((3, 1, 2))(cbam_feature)

    return multiply([input_feature, cbam_feature])


def spatial_att(input_feature):
    kernel_size = 7
    if K.image_data_format() == "channels_first":
        channel = input_feature._keras_shape[1]
        cbam_feature = Permute((2, 3, 1))(input_feature)
    else:
        channel = input_feature._keras_shape[-1]
        cbam_feature = input_feature

    avg_pool = Lambda(lambda x: K.mean(x, axis=4, keepdims=True))(cbam_feature)
    assert avg_pool._keras_shape[-1] == 1
    max_pool = Lambda(lambda x: K.max(x, axis=4, keepdims=True))(cbam_feature)
    assert max_pool._keras_shape[-1] == 1
    concat = Concatenate(axis=4)([avg_pool, max_pool])
    assert concat._keras_shape[-1] == 2
    cbam_feature = Conv3D(filters=1,
                          kernel_size=kernel_size,
                          strides=1,
                          padding='same',
                          activation='sigmoid',
                          kernel_initializer='he_normal',
                          use_bias=False)(concat)
    assert cbam_feature._keras_shape[-1] == 1

    if K.image_data_format() == "channels_first":
        cbam_feature = Permute((3, 1, 2))(cbam_feature)

    return multiply([input_feature, cbam_feature])

def grid3d_model_layer(inputs, num_filters, activation):
    kernel_size = 3
    strides = (1, 1, 1)
    padding = 'same'
    data_format = None
    dilation_rate = (1, 1, 1)
    use_bias = True
    kernel_initializer = 'glorot_uniform'
    bias_initializer = 'zeros'
    kernel_regularizer = None
    bias_regularizer = None
    activity_regularizer = None
    kernel_constraint = None
    bias_constraint = None
    batch_normalization = False
    conv_first = True
    conv = Conv3D(filters=num_filters,kernel_size=kernel_size,strides=strides,padding=padding,data_format=data_format,dilation_rate=dilation_rate,activation=activation,use_bias=use_bias,
                     kernel_initializer=kernel_initializer,bias_initializer=bias_initializer,kernel_regularizer=kernel_regularizer,bias_regularizer=bias_regularizer,
                     activity_regularizer=activity_regularizer,kernel_constraint=kernel_constraint,bias_constraint=bias_constraint)

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x

def loss_tox21(y_true, y_pred):
    is_valid = y_true**2 > 0
    loss_mat = K.binary_crossentropy( (y_true+1)/2, y_pred )
    loss_mat = tf.where( is_valid, loss_mat, tf.zeros_like(loss_mat) )
    loss = tf.reduce_sum(loss_mat) / tf.reduce_sum( tf.cast(is_valid, tf.float32) )
    return loss

def model_3D_grid(hyper):
    units_conv_coord = hyper["units_conv_coord"]
    units_conv_atom = hyper["units_conv_atom"]
    units_dense = hyper["units_dense"]
    chebyshev_order = hyper["chebyshev_order"]
    num_layers = hyper["num_layers"]
    std = hyper["data_std"]
    loss = hyper["loss"]
    task = hyper["task"]
    pooling = hyper["pooling"]
    outputs = hyper["outputs"]
    gridshape = hyper["gridshape"]
    isattention = hyper["isattention"]

    grid =Input(name='grid_inputs', shape=gridshape)

    # 1. grid3d
    x = MaxPooling3D(pool_size=(2, 2, 2),
                     strides=(2, 2, 2),
                     padding='valid',
                     data_format='channels_last',
                     input_shape=gridshape)(grid)
    y = grid3d_model_layer(inputs=x, num_filters=32, activation='relu')
    if isattention:
        y = attention_module_layer(y)
        y = Activation('relu')(y)
    y = Dropout(0.5)(y)

    y = MaxPooling3D(pool_size=(2, 2, 2),
                     strides=(2, 2, 2),
                     padding='valid',
                     data_format='channels_last')(y)
    y = grid3d_model_layer(y, num_filters=64, activation='relu')
    if isattention:
        y = attention_module_layer(y)
        y = Activation('relu')(y)
    y = Dropout(0.5)(y)

    y = MaxPooling3D(pool_size=(2, 2, 2),
                     strides=(2, 2, 2),
                     padding='valid',
                     data_format='channels_last')(y)
    y = grid3d_model_layer(y, num_filters=128, activation='relu')
    if isattention:
        y = attention_module_layer(y)
        y = Activation('relu')(y)
    y = Dropout(0.5)(y)

    y = Flatten()(y)

    if task == "regression":
        out = Dense(outputs, activation='linear', name='output')(y)
        model = Model(inputs=[grid], outputs=out)
        model.compile(optimizer=Adam(lr=0.001), loss=loss, metrics=[std_mae(std=std), std_rmse(std=std)])
    elif task == "binary":
        out = Dense(outputs, activation='sigmoid', name='output')(y)
        model = Model(inputs=[grid], outputs=out)
        model.compile(optimizer=Adam(lr=0.001), loss=loss)

    return model
