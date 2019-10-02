from keras.applications import ResNet50
from keras.models import Model
from keras.layers import (Input, LeakyReLU, BatchNormalization, Add,
                          GlobalMaxPooling2D, Concatenate, GlobalMaxPooling1D,
                          SeparableConv2D, UpSampling2D)
from kernel_regularizers import l1_l2
from .neural_layers import expand_dims

def create_network((input_shape, prior, L1=0, L2=0, dropout=0, KERNEL_NUM=64):
    resnet = ResNet50(include_top=False, input_shape=input_shape)
    stage_names = ['activation_10', 'activation_22', 'activation_40', 'activation_49']
    stages = [resnet.get_layer(stage_name).output for stage_name in stage_names]

    feature_model = Model(inputs=resnet.input, outputs=stages)
    for l in feature_model.layers:
        l.trainable = False

    # Feature Pyramid Network
    inp = Input(input_shape)
    features_at_stages = feature_model(inp)

    x = SeparableConv2D(KERNEL_NUM, 1, padding='same', kernel_regularizer=l1_l2(l1=L1, l2=L2))(features_at_stages[-1])
    x = BatchNormalization()(x)
    P = LeakyReLU(name='P5')(x)
    pyramid_features = [P]

    for i in range(2,5):
        x = SeparableConv2D(KERNEL_NUM, 1, padding='same', kernel_regularizer=l1_l2(l1=L1, l2=L2))(features_at_stages[-i])
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)

        y = UpSampling2D(interpolation='bilinear')(P)
        z = Add()([x,y])

        z = SeparableConv2D(KERNEL_NUM, 3, padding='same', kernel_regularizer=l1_l2(l1=L1, l2=L2))(z)
        z = BatchNormalization()(z)
        P = LeakyReLU(name='P{}'.format(6-i))(z)

        pyramid_features.append(P)

    FPN = Model(inputs=inp, outputs=pyramid_features, name='FPN')

    # Grasp proposal Network
    inp = Input(shape=(None, None, KERNEL_NUM))
    x = SeparableConv2D(KERNEL_NUM, 3)(inp)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    x = SeparableConv2D(512, 1, activation='sigmoid')(x)
    x = GlobalMaxPooling2D()(x)

    GPN = Model(inputs=inp, outputs=x, name='GPN')

    # Trainable model
    inp = Input(shape=input_shape)

    pyramid_features = FPN(inp)
    proposals = []
    for feature_map in pyramid_features:
        x = GPN(feature_map)
        x = expand_dims(axis=1)(x)
        proposals.append(x)
    x = Concatenate(axis=1)(proposals)
    x = GlobalMaxPooling1D()(x)

    model = Model(inputs=inp, outputs=x, name='Final_Model')
    return model
