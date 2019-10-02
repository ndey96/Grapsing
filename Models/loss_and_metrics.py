import keras.backend as K

def all_way_binary_cross_entropy(y_true, y_pred):
    """
    Both y_true and y_pred have shape (batch_size, 8, 8, 8).

    Returns a Keras tensor that is (batch_size, 1)
    """
    y_true = K.batch_flatten(y_true)
    y_pred = K.batch_flatten(y_pred) + K.epsilon()

    ce_mask = K.cast(K.greater(y_true, 0.5), dtype=K.floatx())
    # binary_crossentropy = -(ce_mask*K.log(y_pred) + (1-ce_mask)*K.log(1 - y_pred))

    filter = K.cast(K.greater(y_true, 0), dtype=K.floatx()) # Binary mask
    loss = filter * K.binary_crossentropy(ce_mask, y_pred)
    return K.sum(loss, axis=-1)

def dense_binary_cross_entropy(y_true, y_pred):
    """
    Both y_true and y_pred have shape (batch_size, 8, 8, 8).

    Returns a Keras tensor that is (batch_size, 1)
    """
    y_true = K.batch_flatten(y_true)
    y_pred = K.batch_flatten(y_pred)

    p_t = y_true*y_pred + (1-y_true)*(1-y_pred)

    loss = K.pow((1-p_t), 2)*K.binary_crossentropy(y_true, y_pred)
    return K.sum(loss, axis=-1)

def single_accuracy(y_true, y_pred):
    y_true = K.batch_flatten(y_true)
    y_pred = K.batch_flatten(y_pred)

    ind_true = K.argmax(y_true, axis=-1)
    ind_pred = K.argmax(y_pred, axis=-1)
    matches = K.cast(K.equal(ind_true, ind_pred), dtype=K.floatx())
    #filter = K.cast(K.greater(y_true, 0), dtype=K.floatx())

    #true_class = K.greater(y_true, 0.5)
    #pred_class = K.greater(y_pred, 0.5)
    #matches = K.cast(K.equal(true_class, pred_class), dtype=K.floatx())

    #matches = K.sum(filter*matches,axis=-1)
    return K.mean(matches)
