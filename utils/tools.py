import tensorflow as tf

def avg_pool(x, ksize, strides, name, padding='SAME'):
    return tf.nn.avg_pool(x, ksize=ksize, strides=strides, name=name, padding=padding)

def max_pool(x, ksize, strides, name, padding='SAME'):
    return tf.nn.max_pool(x, ksize=ksize, strides=strides, name=name, padding=padding)

def conv_layer(x, conv_data, name, reuse=None):
    with tf.variable_scope(name):

