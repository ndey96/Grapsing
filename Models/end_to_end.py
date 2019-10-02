from keras.models import Model
import keras.backend as K
from keras.layers import (Conv2D, LeakyReLU, Input, BatchNormalization, Add,
                          Lambda, concatenate, Conv3D, Dropout, Activation)
from keras.regularizers import l1_l2

from .neural_layers import residual_block, bilinear_resize, expand_dims

def create_network(input_shape, prior, L1=0, L2=0, dropout=0, KERNEL_NUM = 64):
    inp = Input(shape=input_shape)
    x = Conv2D(KERNEL_NUM, 7, padding='same', strides=2, kernel_regularizer=l1_l2(l1=L1, l2=L2))(inp)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Dropout(dropout)(x)

    x = Conv2D(KERNEL_NUM, 3, padding='same', strides=2, kernel_regularizer=l1_l2(l1=L1, l2=L2))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Dropout(dropout)(x)

    # First Residual Block
    x = residual_block(x, bottleneck_kernels=KERNEL_NUM//4,
                               out_kernels=KERNEL_NUM,
                               kernel_size=3,
                               identity=False,
                               L1=L1,
                               L2=L2)
    x = Dropout(dropout)(x)
    for i in range(2):
        x = residual_block(x, bottleneck_kernels=KERNEL_NUM//4,
                                    out_kernels=KERNEL_NUM,
                                    kernel_size=3,
                                    identity=True,
                                    L1=L1,
                                    L2=L2)
        x = Dropout(dropout)(x)

    # Downsampling
    x = Conv2D(4*KERNEL_NUM, 3, padding='same', strides=2, kernel_regularizer=l1_l2(l1=L1, l2=L2))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Dropout(dropout)(x)

    # Second Residual Block
    x = residual_block(x, bottleneck_kernels=KERNEL_NUM//2,
                               out_kernels=2*KERNEL_NUM,
                               kernel_size=3,
                               identity=False,
                               L1=L1,
                               L2=L2)
    x = Dropout(dropout)(x)
    for i in range(3):
        x = residual_block(x, bottleneck_kernels=KERNEL_NUM//2,
                                    out_kernels=2*KERNEL_NUM,
                                    kernel_size=3,
                                    identity=True,
                                    L1=L1,
                                    L2=L2)
        x = Dropout(dropout)(x)

    # Downsampling
    x = Conv2D(2*KERNEL_NUM, 3, padding='same', strides=2, kernel_regularizer=l1_l2(l1=L1, l2=L2))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Dropout(dropout)(x)

    # Third Residual Block
    x = residual_block(x, bottleneck_kernels=KERNEL_NUM,
                               out_kernels=4*KERNEL_NUM,
                               kernel_size=3,
                               identity=False,
                               L1=L1,
                               L2=L2)
    x = Dropout(dropout)(x)
    for i in range(5):
        x = residual_block(x, bottleneck_kernels=KERNEL_NUM,
                                    out_kernels=4*KERNEL_NUM,
                                    kernel_size=3,
                                    identity=True,
                                    L1=L1,
                                    L2=L2)
        x = Dropout(dropout)(x)

    # Reshape to (None, 8,8,8, 1024)
    x = bilinear_resize((8,8))(x)
    x = expand_dims(axis=3)(x)
    repeat = Lambda( lambda x: K.tile( x, (1, 1, 1, 7, 1) ))(x)
    x = concatenate([x,repeat], axis=3)

    # Trainable residual 3D blocks
    x = residual_block(x, bottleneck_kernels=2*KERNEL_NUM,
                                out_kernels=8*KERNEL_NUM,
                                kernel_size=3,
                                identity=False,
                                conv=Conv3D,
                                L1=L1,
                                L2=L2)
    x = Dropout(dropout)(x)

    for i in range(2): # Originally 2
        x = residual_block(x, bottleneck_kernels=2*KERNEL_NUM,
                                   out_kernels=8*KERNEL_NUM,
                                   kernel_size=3,
                                   identity=True,
                                   conv=Conv3D,
                                   L1=L1,
                                   L2=L2)
    x = Dropout(dropout)(x)

    x = Conv3D(1, 3, padding='same', activation='tanh')(x)
    x = Lambda(lambda x: K.squeeze(x, axis=-1))(x)

    def add_prior(x, prior=prior):
        prior = K.constant(K.cast_to_floatx(prior))
        prior = expand_dims(axis=0)(prior)
        prior = Lambda( lambda prior: K.tile( prior, (K.shape(x)[0], 1, 1, 1) ))(prior)
        return x + prior

    x = Lambda( lambda x: add_prior(x) )(x)
    return Model(inp, x)
