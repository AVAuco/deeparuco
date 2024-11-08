import tensorflow as tf

def weighted_loss(y_true, y_pred):

    error = tf.math.square(y_true - y_pred)
    y_norm = tf.math.divide(y_true - tf.math.reduce_min(y_true), 
                            tf.math.reduce_max(y_true) - tf.math.reduce_min(y_true))
    w_mse = tf.reduce_mean(tf.math.multiply(error, 9 * y_norm + 1))

    return w_mse