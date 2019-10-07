from keras.utils import multi_gpu_model
from keras.applications import resnet50
from keras.callbacks import CSVLogger, ModelCheckpoint, ReduceLROnPlateau
import tensorflow as tf
import numpy as np
from numpy.random import randint, uniform
from rotation_helpers import perturb_xyz


def prepare_model(create_network,
                  input_shape,
                  loss,
                  optimizer,
                  metrics,
                  prior,
                  GPUs=1,
                  L1=0,
                  L2=0,
                  dropout=0,
                  kernel_num=64):
    if GPUs == 1:
        template_model = create_network(
            input_shape,
            prior,
            L1=L1,
            L2=L2,
            dropout=dropout,
            KERNEL_NUM=kernel_num)
        model = template_model
    elif GPUs > 1:
        with tf.device('/cpu:0'):
            template_model = create_network(
                input_shape,
                prior,
                L1=L1,
                L2=L2,
                dropout=dropout,
                KERNEL_NUM=kernel_num)
            model = multi_gpu_model(template_model, gpus=GPUs)
    else:
        raise ValueError(
            'GPUs needs to be an integer greater than or equal to 1, not {}'.
            format(GPUs))

    template_model.summary()
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    return (template_model, model)


def train_generator(batch_size, train_images, train_poses, train_targets):
    total_num = train_targets.shape[0]
    shuffled_indices = np.arange(total_num)

    # For augmentation bin 2: min/max of xyz with successful grips
    xmin, xmax = -0.11448083617113874, 0.02998685945970189
    ymin, ymax = -0.16689919684337934, 0.07206119878809979
    zmin, zmax = 0.4016680300406181, 0.5670741942050048
    x_range = xmax - xmin
    y_range = ymax - ymin
    z_range = zmax - zmin

    xyz_min_perturbs = (0.06, 0.1, 0.03)
    while True:
        np.random.shuffle(shuffled_indices)
        for i in range(total_num // batch_size):
            current_indices = shuffled_indices[i * batch_size:(
                i + 1) * batch_size]
            batch_images = train_images[current_indices].copy()
            batch_poses = train_poses[current_indices].copy()
            batch_targets = train_targets[current_indices].copy()

            # Generate synthetic negative examples
            augmentation_bins = randint(4, size=batch_size)
            for idx, aug_bin in enumerate(augmentation_bins):
                # aug_bin == 0 use original data
                # Rotate pose orientation by rand(pi/4, pi) and mark as negative
                if aug_bin == 1:
                    rot_mag = uniform(np.pi / 4, np.pi)
                    rot_sign = np.random.choice([-1, 1], 1)[0]
                    batch_poses[idx][6] = (
                        batch_poses[idx][6] - rot_sign * rot_mag) % (2 * np.pi)
                    batch_targets[idx] = 0
                # Shift xyz to be outside the range of the successful grip values and mark as negative
                if aug_bin == 2:
                    x_shift = [
                        uniform(xmin - 2 * x_range, xmin),
                        uniform(xmax, xmax + 2 * x_range)
                    ][randint(2)]
                    y_shift = [
                        uniform(ymin - 2 * y_range, ymin),
                        uniform(ymax, ymax + 2 * y_range)
                    ][randint(2)]
                    z_shift = [
                        uniform(zmin - 2 * z_range, zmin),
                        uniform(zmax, zmax + 2 * z_range)
                    ][randint(2)]
                    batch_poses[idx][0:3] = [x_shift, y_shift, z_shift]
                    batch_targets[idx] = 0
                # Perturb xyz in gripper coordinates based on robustness analysis and mark as negative
                if aug_bin == 3:
                    xyz_perturbations = np.array([
                        uniform(xyz_min_perturbs[0], 3 * xyz_min_perturbs[0]),
                        uniform(xyz_min_perturbs[1], 3 * xyz_min_perturbs[1]),
                        uniform(xyz_min_perturbs[2], 3 * xyz_min_perturbs[2])
                    ])
                    batch_poses[idx][0:3] = perturb_xyz(batch_poses[idx],
                                                        xyz_perturbations)
                    batch_targets[idx] = 0

            yield {
                'depth_input': batch_images,
                'pose_input': batch_poses
            }, {
                'final_output': batch_targets
            }


def val_generator(batch_size, val_images, val_poses, val_targets):
    total_num = val_targets.shape[0]
    shuffled_indices = np.arange(total_num)
    while True:
        # yield {
        #           'depth_input': val_images,
        #           'pose_input': val_poses
        #       }, {
        #           'final_output': val_targets
        #       }
        np.random.shuffle(shuffled_indices)
        for i in range(total_num // batch_size):
            current_indices = shuffled_indices[i * batch_size:(
                i + 1) * batch_size]
            batch_images = val_images[current_indices]
            batch_poses = val_poses[current_indices]
            batch_targets = val_targets[current_indices]

            yield {
                'depth_input': batch_images,
                'pose_input': batch_poses
            }, {
                'final_output': batch_targets
            }


def load_data(train_set, val_set):
    print('\tLoading Poses')
    val_poses = np.load('Data/pos_val_poses.npy')
    train_poses = np.load('Data/pos_train_poses.npy')

    print('\tLoading Targets')
    val_targets = np.ones((val_poses.shape[0], 1))
    train_targets = np.ones((train_poses.shape[0], 1))

    print('\tLoading Images')
    val_images = np.load('Data/depth_validation_image_data.npy')
    train_images = np.load('Data/depth_training_image_data.npy')

    val_images = np.reshape(val_images, (*val_images.shape, 1))
    train_images = np.reshape(train_images, (*train_images.shape, 1))

    return train_images, val_images, train_poses, val_poses, train_targets, val_targets


def load_network(arch):
    if arch == 'cnn_pose_img':
        from Models.cnn_pose_img import create_network
    elif arch == 'fcn_pose':
        from Models.fcn_pose import create_network
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
    opt_name = config['optimizer']['optimizer']
    learning_rate = config['optimizer'].getfloat('learning_rate')
    clipnorm = config['optimizer'].getfloat('clipnorm')
    momentum = config['optimizer'].getfloat('momentum')
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
    DROPOUT_RATE = config['regularizer'].getfloat('dropout_rate')

    # Callbacks
    callbacks = []
    if config['callbacks'].getboolean('csv_logger'):
        csv_fname = config['callbacks']['csv_fname']
        csv_logger = CSVLogger(csv_fname)
        callbacks.append(csv_logger)

    if config['callbacks'].getboolean('model_checkpoint'):
        model_fname = config['callbacks']['model_fname']
        save_best = config['callbacks'].getboolean('model_save_best')
        model_checkpoint = ModelCheckpoint(
            model_fname, save_best_only=save_best, save_weights_only=True)
        callbacks.append(model_checkpoint)

    if config['callbacks'].getboolean('reduce_lr_on_plateau'):
        monitor = config['callbacks']['reduce_lr_monitor']
        factor = float(config['callbacks']['reduce_lr_factor'])
        patience = config['callbacks'].getfloat('reduce_lr_patience')
        min_delta = float(config['callbacks'].getfloat('reduce_lr_min_delta'))
        reduce_lr_on_plateau = ReduceLROnPlateau(
            monitor=monitor,
            factor=factor,
            patience=patience,
            min_delta=min_delta)
        callbacks.append(reduce_lr_on_plateau)

    # Dataset names
    train_set = config['training']['training_set']
    val_set = config['training']['validation_set']

    # Hyperparameters
    BATCH_SIZE = config['training'].getint('batch_size')
    val_set = config['training']['validation_set']
    train_set = config['training']['training_set']

    print('Building Network')
    from keras.metrics import binary_accuracy
    from keras.losses import binary_crossentropy
    (template, model) = prepare_model(
        create_network,
        input_shape=[224, 224, 3],
        loss=binary_crossentropy,
        optimizer=opt,
        metrics=[binary_accuracy],
        GPUs=1,
        L1=L1_REGULARIZER,
        L2=L2_REGULARIZER,
        dropout=DROPOUT_RATE,
        prior=None,
        kernel_num=kernel_num)

    print('Loading Data')
    train_images, val_images, train_poses, val_poses, train_targets, val_targets = load_data(
        train_set, val_set)

    # cutoff = 255
    # train_images = train_images[:cutoff]
    # train_poses = train_poses[:cutoff]
    # train_targets = train_targets[:cutoff]

    print('Configuring Generator')
    train_gen = train_generator(BATCH_SIZE, train_images, train_poses,
                                train_targets)

    val_gen = val_generator(BATCH_SIZE, val_images, val_poses, val_targets)

    print('Beginning Training')
    model.fit_generator(
        train_gen,
        steps_per_epoch=train_targets.shape[0] / BATCH_SIZE,
        epochs=1000,
        validation_data=val_gen,
        validation_steps=val_targets.shape[0] / BATCH_SIZE,
        callbacks=[csv_logger, model_checkpoint, reduce_lr_on_plateau])
