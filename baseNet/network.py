import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim import losses
from tensorflow.contrib.slim import arg_scope

import numpy as np

class network(object):
    def __init__(self):
        self._predictions = {}
        self._losses = {}
        self._layers = {}
        self._score_summaries = {}
        self._train_summaries = []

    def _avg_pool(self, x, name, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME'):
        return tf.nn.avg_pool(x, ksize=ksize, strides=strides, name=name, padding=padding)

    def _max_pool(self, x, name, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME'):
        return tf.nn.max_pool(x, ksize=ksize, strides=strides, name=name, padding=padding)

    def _conv_layer(self, x, conv_data, name, activation='relu', strides=[1, 1, 1, 1], padding='SAME', is_training=False, reuse=None):
        with tf.variable_scope(name):
            kernel = tf.constant(conv_data[0], name='weights')
            biases = tf.constant(conv_data[1], name='biases'
            conv = tf.nn.conv2d(x, kernel, strides=strides, padding=padding)
            bias = tf.nn.bias_add(conv, biases)
            if activation=='relu':
                return tf.nn.relu(bias)
            elif activation==None:
                return bias
            return None

    def _dropout_layer(self, x, name, keep_prob=0.5):
        return tf.nn.dropout(x, keep_prob, name=name)

    def _fc_layer(self, x, fc_data, name, is_training=False, reuse=None):
        with tf.variable_scope(name):
            shape = x.get_shape().as_list()
            dim = 1
            for d in shape[1:]:
                dim *= d
            x = tf.reshape(x, [-1, dim])
            weights = tf.constant(fc_data[0], name='weights')
            biases = tf.constant(fc_data[1], name='biases')
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)
            return fc
