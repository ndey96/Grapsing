from keras.models import Model
import keras.backend as K
from keras.layers import (Conv2D, LeakyReLU, Input, BatchNormalization, Add,
                          Lambda, concatenate, Conv3D, Dropout, Activation,
                          MaxPooling2D, Flatten, Dense)
from keras.regularizers import l1_l2


def create_network(input_shape, prior, L1=0, L2=0, dropout=0, KERNEL_NUM=64):
    pose_input = Input(shape=(7,), dtype='float32', name='pose_input')
    x = Dense(64, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.25)(x)
    final_output = Dense(1, activation='sigmoid', name='final_output')(x)

    model = Model(inputs=[depth_input, pose_input], outputs=[final_output])
    # model.compile(optimizer='rmsprop', loss='binary_crossentropy')
    # print(model.summary())
    return model