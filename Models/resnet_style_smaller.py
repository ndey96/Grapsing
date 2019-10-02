from keras.applications import ResNet50
from keras.models import Model
import keras.backend as K
from keras.layers import (Conv2D, LeakyReLU, Input, BatchNormalization, Add,
                          MaxPooling2D, Lambda, concatenate, Conv3D, Dropout,
                          Activation)
from keras.regularizers import l1_l2

from .neural_layers import residual_block, bilinear_resize, expand_dims

def create_network(input_shape, prior, L1=0, L2=0, dropout=0, KERNEL_NUM=64):
    resnet = ResNet50(include_top=False, input_shape=input_shape)
    feature_model = Model(resnet.input, resnet.layers[80].output)
    for l in feature_model.layers:
        l.trainable = False

    inp = Input(shape=input_shape)
    features = feature_model(inp)

    # Downsampling
    x = Conv2D(1, 3, padding='same', strides=2, kernel_regularizer=l1_l2(l1=L1, l2=L2))(features)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    x = Dropout(dropout)(x)

    # Trainable residual blocks
    # x = residual_block(x, bottleneck_kernels=KERNEL_NUM//4,
    #                            out_kernels=KERNEL_NUM,
    #                            kernel_size=3,
    #                            identity=False,
    #                            L1=L1,
    #                            L2=L2)
    # x = Dropout(dropout)(x)
    # for i in range(1): # Originally 5
    #     x = residual_block(x, bottleneck_kernels=KERNEL_NUM//4,
    #                                 out_kernels=KERNEL_NUM,
    #                                 kernel_size=3,
    #                                 identity=True,
    #                                 L1=L1,
    #                                 L2=L2)
    #     x = Dropout(dropout)(x)

    # Reshape to (None, 8,8,8, 1024)
    x = bilinear_resize((8,8))(x)
    x = expand_dims(axis=1)(x)
    repeat = Lambda( lambda x: K.tile( x, (1, 7, 1, 1, 1) ))(x)
    x = concatenate([x,repeat], axis=1)

    # Trainable residual 3D blocks
    x = residual_block(x, bottleneck_kernels=KERNEL_NUM//4,
                                out_kernels=KERNEL_NUM,
                                kernel_size=3,
                                identity=False,
                                conv=Conv3D,
                                L1=L1,
                                L2=L2)
    x = Dropout(dropout)(x)

    # for i in range(1): # Originally 2
    #     x = residual_block(x, bottleneck_kernels=KERNEL_NUM//4,
    #                                out_kernels=KERNEL_NUM,
    #                                kernel_size=3,
    #                                identity=True,
    #                                conv=Conv3D,
    #                                L1=L1,
    #                                L2=L2)
    # x = Dropout(dropout)(x)

    x = Conv3D(1, 3, padding='same', activation='tanh')(x)
    x = Lambda(lambda x: K.squeeze(x, axis=-1))(x)

    def add_prior(x, prior=prior):
        prior = K.constant(K.cast_to_floatx(prior))
        prior = expand_dims(axis=0)(prior)
        prior = Lambda( lambda prior: K.tile( prior, (K.shape(x)[0], 1, 1, 1) ))(prior)
        return x + prior

    x = Lambda( lambda x: add_prior(x) )(x)
    return Model(inp, x)
