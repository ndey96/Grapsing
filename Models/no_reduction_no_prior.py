from keras.applications import ResNet50
from keras.models import Model
import keras.backend as K
from keras.layers import (Conv2D, LeakyReLU, Input, BatchNormalization, Add,
                          MaxPooling2D, Lambda, concatenate, Conv3D, Dropout,
                          Activation, Flatten, Dense, GlobalAveragePooling2D)
from keras.regularizers import l1_l2

from .neural_layers import residual_block, bilinear_resize, expand_dims, residual_stage

def create_network(input_shape, prior, L1=0, L2=0, dropout=0, KERNEL_NUM=64):
    inp = Input(shape=input_shape)

    x = Conv2D(KERNEL_NUM, 5, padding='same', kernel_regularizer=l1_l2(l1=L1, l2=L2))(inp)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    x = Conv2D(2*KERNEL_NUM, 5, padding='same', kernel_regularizer=l1_l2(l1=L1, l2=L2))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    x = Conv2D(4*KERNEL_NUM, 5, padding='same', kernel_regularizer=l1_l2(l1=L1, l2=L2))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    x = Conv2D(8*KERNEL_NUM, 3, padding='same', kernel_regularizer=l1_l2(l1=L1, l2=L2))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    x = Conv2D(16*KERNEL_NUM, 3, padding='same', kernel_regularizer=l1_l2(l1=L1, l2=L2))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    x = Conv2D(KERNEL_NUM, 28, strides=28, kernel_regularizer=l1_l2(l1=L1, l2=L2))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    x = Conv2D(8, 1, activation='sigmoid', kernel_regularizer=l1_l2(l1=L1, l2=L2))(x)
    return Model(inp, x)
