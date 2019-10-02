from keras.utils import multi_gpu_model
from keras.applications import resnet50
from keras.callbacks import CSVLogger, ModelCheckpoint, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import numpy as np

def prepare_model(create_network, input_shape, loss, optimizer, metrics, prior, GPUs=1, L1=0, L2=0, dropout=0, kernel_num=64):
    if GPUs==1:
        template_model = create_network(input_shape, prior, L1=L1, L2=L2, dropout=dropout, KERNEL_NUM=kernel_num)
        model = template_model
    elif GPUs>1:
        with tf.device('/cpu:0'):
            template_model = create_network(input_shape, prior, L1=L1, L2=L2, dropout=dropout, KERNEL_NUM=kernel_num)
            model = multi_gpu_model(template_model, gpus=GPUs)
    else:
        raise ValueError('GPUs needs to be an integer greater than or equal to 1, not {}'.format(GPUs))

    template_model.summary()
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    return (template_model, model)

def data_generator(batch_size, imgs, pos_cubes):
    total_num = imgs.shape[0]
    shuffled_indeces = np.arange(total_num)
    prep = ImageDataGenerator(width_shift_range=20, height_shift_range=20, zoom_range=0.1, rotation_range=22)
    while True:
        np.random.shuffle(shuffled_indeces)
        for i in range(total_num):
            current_indeces = shuffled_indeces[i*batch_size:(i+1)*batch_size]
            current_images = imgs[current_indeces]
            if current_images.shape[0]!=batch_size:
                continue

            batch_images = np.zeros_like(current_images)
            for j in range(batch_images.shape[0]):
                trans_img = prep.random_transform(current_images[j])
                batch_images[j] = trans_img

            batch_cubes = np.zeros([current_indeces.shape[0],8,8,8])
            for j,cur_ind in enumerate(current_indeces):
                pz,py,px = pos_cubes[cur_ind]
                batch_cubes[j, px,py,pz] = 1

            yield batch_images, batch_cubes

def load_data(train_set, val_set):
    print('\tLoading Targets')
    train_pos_cubes = np.load('Data/for_training/compact/training_positive_cubes.npy').astype(np.int32)[:, 1:]
    val_pos_cubes = np.load('Data/for_training/compact/validation_positive_cubes.npy').astype(np.int32)[:, 1:]

    print('\tLoading Images')
    #train_images = np.zeros([31062,224,224,3], dtype=np.int8)
    if train_set=='small':
        train_images = np.load('Data/for_training/compact/small_image_set.npy')
        train_pos_cubes = train_pos_cubes[:512]
        train_neg_cubes = train_neg_cubes[:512]
    elif train_set=='full':
        train_images = np.load('Data/for_training/compact/training_image_data.npy')
    else:
        raise ValueError('{} is not a valid training_set'.format(train_set))

    val_images = np.load('Data/for_training/compact/validation_image_data.npy')

    if val_set=='random':
        val_images = val_images[571:]
        val_pos_cubes = val_pos_cubes[571:]
        total_num = 1000-571
    elif val_set=='object':
        val_images = val_images[:571]
        val_pos_cubes = val_pos_cubes[:571]
        total_num = 571
    elif val_set=='full':
        total_num = 1000
    else:
        raise ValueError('{} is not a valid validation_set'.format(val_set))

    print('\tPreprocessing Images')
    train_images = resnet50.preprocess_input(train_images)
    val_images = resnet50.preprocess_input(val_images)

    print('\tProcessing Validation Set')
    validation_cubes = np.zeros([total_num, 8,8,8])
    for i in range(total_num):
        pz,py,px = val_pos_cubes[i]
        validation_cubes[i, px,py,pz] = 1

    return (train_images, val_images), train_pos_cubes, validation_cubes

def load_network(arch):
    if arch == 'resnet_style':
        from Models.resnet_style import create_network
    elif arch == 'resnet_style_smaller':
            from Models.resnet_style_smaller import create_network
    elif arch == 'resnet_style_smaller_split':
            from Models.resnet_style_smaller_split import create_network
    elif arch == 'end_to_end':
        from Models.end_to_end import create_network
    elif arch == 'end_to_end_sep':
        from Models.end_to_end_sep import create_network
    elif arch == 'simplest':
        from Models.simplest import create_network
    elif arch == 'simplest_2':
        from Models.simplest_2 import create_network
    elif arch == 'resnet_style_smaller_split_no_prior':
        from Models.resnet_style_smaller_split_no_prior import create_network
    elif arch == 'simple_smaller_split_no_prior':
        from Models.simple_smaller_split_no_prior import create_network
    elif arch == 'no_reduction_no_prior':
        from Models.no_reduction_no_prior import create_network
    else:
        raise ValueError('{} is not a valid architecture'.format(arch))
    return create_network

if __name__ == '__main__':
    import configparser
    config = configparser.ConfigParser()
    config.read('training_config.ini')

    # Architecture
    arch = config['architecture']['architecture']
    kernel_num = config['architecture'].getint('kernel_num')
    create_network = load_network(arch)

    # Optimizer
    opt_name      = config['optimizer']['optimizer']
    learning_rate = config['optimizer'].getfloat('learning_rate')
    clipnorm      = config['optimizer'].getfloat('clipnorm')
    momentum      = config['optimizer'].getfloat('momentum')
    if opt_name == 'adam':
        from keras.optimizers import Adam
        opt = Adam(lr=learning_rate, clipnorm=clipnorm)
    elif opt_name == 'sgd':
        from keras.optimizers import SGD
        opt = SGD(lr=learning_rate, clipnorm=clipnorm, momentum=momentum)
    else:
        raise ValueError('{} is not a valid optimizer'.format(opt_name))

    # Regularizer
    L1_REGULARIZER = config['regularizer'].getfloat('l1_penalty')
    L2_REGULARIZER = config['regularizer'].getfloat('l2_penalty')
    DROPOUT_RATE   = config['regularizer'].getfloat('dropout_rate')

    # Callbacks
    callbacks = []
    if config['callbacks'].getboolean('csv_logger'):
        csv_fname  = config['callbacks']['csv_fname']
        csv_logger = CSVLogger(csv_fname)
        callbacks.append(csv_logger)

    if config['callbacks'].getboolean('model_checkpoint'):
        model_fname = config['callbacks']['model_fname']
        save_best   = config['callbacks'].getboolean('model_save_best')
        model_checkpoint = ModelCheckpoint(model_fname, save_best_only=save_best)
        callbacks.append(model_checkpoint)

    if config['callbacks'].getboolean('reduce_lr_on_plateau'):
        monitor   = config['callbacks']['reduce_lr_monitor']
        factor    = config['callbacks']['reduce_lr_factor']
        patience  = config['callbacks'].getfloat('reduce_lr_patience')
        min_delta = config['callbacks'].getfloat('reduce_lr_min_delta')
        reduce_lr_on_plateau = ReduceLROnPlateau(monitor=monitor,
                                                 factor=factor,
                                                 patience=patience,
                                                 min_delta=min_delta)
        callbacks.append(reduce_lr_on_plateau)

    # Dataset names
    train_set = config['training']['training_set']
    val_set   = config['training']['validation_set']

    # Hyperparameters
    BATCH_SIZE = config['training'].getint('batch_size')
    val_set    = config['training']['validation_set']
    train_set  = config['training']['training_set']


    print('Building Network')
    from Models.loss_and_metrics import single_accuracy, dense_binary_cross_entropy

    (template, model) = prepare_model(create_network, input_shape=[224,224,3],
                                                      loss=dense_binary_cross_entropy,
                                                      optimizer=opt,
                                                      metrics=[single_accuracy],
                                                      GPUs=1,
                                                      L1=L1_REGULARIZER,
                                                      L2=L2_REGULARIZER,
                                                      dropout=DROPOUT_RATE,
                                                      prior=np.load('Data/for_training/prior_compact.npy'),
                                                      kernel_num=kernel_num)

    print('Loading Data')
    images, cubes, validation_cubes = load_data(train_set, val_set)
    train_images, val_images = images
    train_pos_cubes = cubes

    print('Configuring Generator')
    datagen = data_generator(BATCH_SIZE, train_images, train_pos_cubes)

    print('Beginning Training')
    batch_per_epoch = (train_images.shape[0])/BATCH_SIZE
    model.fit_generator(datagen, steps_per_epoch=batch_per_epoch,
                                 epochs=1000,
                                 validation_data=(val_images,validation_cubes),
                                 callbacks=[csv_logger, model_checkpoint])#, reduce_lr_on_plateau])
