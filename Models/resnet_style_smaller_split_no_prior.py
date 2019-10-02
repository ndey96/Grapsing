from keras.applications import ResNet50
from keras.models import Model
import keras.backend as K
from keras.layers import (Conv2D, LeakyReLU, Input, BatchNormalization, Add,
                          MaxPooling2D, Lambda, concatenate, Conv3D, Dropout,
                          Activation)
from keras.regularizers import l1_l2

from .neural_layers import residual_block, bilinear_resize, expand_dims, residual_stage

def create_network(input_shape, prior, L1=0, L2=0, dropout=0, KERNEL_NUM=64):
    resnet = ResNet50(include_top=False, input_shape=input_shape)
    feature_model = Model(resnet.input, resnet.layers[80].output)
    for l in feature_model.layers:
        l.trainable = False

    inp = Input(shape=input_shape)
    features = feature_model(inp)

    # Downsampling
    x = MaxPooling2D()(features)
    #Conv2D(512, 3, padding='same', strides=2, kernel_regularizer=l1_l2(l1=L1, l2=L2))(features)
    #x = BatchNormalization()(x)
    #x = LeakyReLU()(x)

    x = Dropout(dropout)(x)

    # Trainable residual blocks
    start = x
    features_at_depth = []
    for i in range(8):
        x = Conv2D(KERNEL_NUM//8, 1, padding='same', kernel_regularizer=l1_l2(l1=L1, l2=L2))(start)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)

        x = residual_stage(x, KERNEL_NUM//8, iter=5, L1=L1, L2=L2, dropout=dropout, initial_block_is_identity=True)
        x = bilinear_resize((8,8))(x)
        x = expand_dims(axis=3)(x)
        features_at_depth.append(x)
    x = concatenate(features_at_depth, axis=3)

    # Trainable residual 3D blocks
    x = residual_block(x, bottleneck_kernels=KERNEL_NUM//4,
                                out_kernels=KERNEL_NUM,
                                kernel_size=3,
                                identity=False,
                                conv=Conv3D,
                                L1=L1,
                                L2=L2)
    x = Dropout(dropout)(x)

    x = Conv3D(1, 3, padding='same', activation='sigmoid')(x)
    x = Lambda(lambda x: K.squeeze(x, axis=-1))(x)

    return Model(inp, x)
