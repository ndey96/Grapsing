from keras.models import Model
import keras.backend as K
from keras.layers import (Conv2D, LeakyReLU, Input, BatchNormalization, Add,
                          Lambda, concatenate, Conv3D, Dropout, Activation,
                          MaxPooling2D, Flatten, Dense)
from keras.regularizers import l1_l2


def create_network(input_shape, prior, L1=0, L2=0, dropout=0, KERNEL_NUM=64):
    depth_input = Input(
        shape=(
            224,
            224,
            1,
        ), dtype='float32', name='depth_input')

    x = Conv2D(
        32, (3, 3), padding='same', input_shape=depth_input.shape)(depth_input)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(32, (3, 3))(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)

    x = Conv2D(64, (3, 3), padding='same')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(64, (3, 3))(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)

    cnn_output = Flatten()(x)

    pose_input = Input(shape=(7,), dtype='float32', name='pose_input')
    x = concatenate([cnn_output, pose_input])
    x = Dense(64, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.25)(x)
    final_output = Dense(1, activation='sigmoid', name='final_output')(x)

    model = Model(inputs=[depth_input, pose_input], outputs=[final_output])
    # model.compile(optimizer='rmsprop', loss='binary_crossentropy')
    # print(model.summary())
    return model