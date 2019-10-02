from keras.models import Model
import keras.backend as K
from keras.layers import (Conv2D, LeakyReLU, Input, BatchNormalization, Add,
                          Lambda, concatenate, Conv3D, Dropout, Activation)
from keras.regularizers import l1_l2

from .neural_layers import residual_block, bilinear_resize, expand_dims, residual_stage

def create_network(input_shape, prior, L1=0, L2=0, dropout=0, KERNEL_NUM = 64):
    inp = Input(shape=input_shape)
    depth = Lambda(lambda x: x[:,:,:,2],  name='extract_depth')(inp)
    depth = expand_dims(axis=-1)(depth)
    color = Lambda(lambda x: x[:,:,:,:2], name='extract_color')(inp)

    half_kernel = KERNEL_NUM//2
    # Depth Path
    x = Conv2D(half_kernel, 7, padding='same', strides=2, kernel_regularizer=l1_l2(l1=L1, l2=L2))(depth)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Dropout(dropout)(x)

    x = Conv2D(half_kernel, 3, padding='same', strides=2, kernel_regularizer=l1_l2(l1=L1, l2=L2))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Dropout(dropout)(x)

    # First Residual Stage
    x = residual_stage(x, half_kernel, iter=2, L1=L1, L2=L2, dropout=dropout)

    # Downsampling
    x = Conv2D(half_kernel, 3, padding='same', strides=2, kernel_regularizer=l1_l2(l1=L1, l2=L2))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Dropout(dropout)(x)

    # Second Residual Stage
    x = residual_stage(x, 2*half_kernel, iter=3, L1=L1, L2=L2, dropout=dropout)

    # Downsampling
    x = Conv2D(2*half_kernel, 3, padding='same', strides=2, kernel_regularizer=l1_l2(l1=L1, l2=L2))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Dropout(dropout)(x)

    depth = x

    # Color Path
    x = Conv2D(half_kernel, 7, padding='same', strides=2, kernel_regularizer=l1_l2(l1=L1, l2=L2))(color)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Dropout(dropout)(x)

    x = Conv2D(half_kernel, 3, padding='same', strides=2, kernel_regularizer=l1_l2(l1=L1, l2=L2))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Dropout(dropout)(x)

    # First Residual Stage
    x = residual_stage(x, half_kernel, iter=2, L1=L1, L2=L2, dropout=dropout)

    # Downsampling
    x = Conv2D(half_kernel, 3, padding='same', strides=2, kernel_regularizer=l1_l2(l1=L1, l2=L2))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Dropout(dropout)(x)

    # Second Residual Stage
    x = residual_stage(x, 2*half_kernel, iter=3, L1=L1, L2=L2, dropout=dropout)

    # Downsampling
    x = Conv2D(2*half_kernel, 3, padding='same', strides=2, kernel_regularizer=l1_l2(l1=L1, l2=L2))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Dropout(dropout)(x)

    color = x

    # Combine
    combined = concatenate([color, depth], axis=-1)

    # Third Residual Stage
    features_at_depth = []
    for i in range(8):
        x = residual_stage(combined, half_kernel, iter=5, L1=L1, L2=L2, dropout=dropout)
        x = bilinear_resize((8,8))(x)
        x = expand_dims(axis=1)(x)
        features_at_depth.append(x)
    x = concatenate(features_at_depth, axis=1)

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
