import numpy as np
import tensorflow as tf


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


if __name__ == '__main__':
    image = np.random.rand(1, 124, 16, 3)
    # The input x will consist of a tensor of floating point numbers
    x = tf.placeholder(tf.float32, shape=[None, 124, 16, 3])

    W_conv1 = weight_variable([5, 5, 3, 32])
    # a bias vector with a component for each output channel
    b_conv1 = bias_variable([32])
    # Apply the layer by convolving x with the weight tensor, add the bias, apply the ReLU function, and finally max
    # pool.
    h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    # Second Convolutional Layer.
    # Stacks a second layer that provides 64 features for each 5x5 patch
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    W_fc1 = weight_variable([31 * 4 * 64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 31 * 4 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    with sess.as_default():
        print h_pool2_flat.shape
        print h_fc1.shape
