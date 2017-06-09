import json
import logging
import sys

import numpy as np
import tensorflow as tf

from data_tools.batch_manager import BatchManager
from data_tools.data_saver import DataSaver
from utils.logging_utils import logging_reconfig


logging_reconfig()


def train(input_uid):
    def weight_variable(shape, name):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial, name=name)

    def bias_variable(shape, name):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial, name=name)

    def conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    bm = BatchManager().load(input_uid)
    # The input x will consist of a tensor of floating point numbers of shape (?, 124, 32, 3)
    x = tf.placeholder(tf.float32, shape=[None, bm.n_channels, bm.trial_size, bm.n_comps])

    result = dict()

    # First Convolutional Layer.
    # It will compute 32 features for each 5x5 patch. Its weight tensor will have a shape of
    # [5, 5, 3, 32]. The first two dimensions are the patch size, the next is the number of electric field components,
    # and the last is the number of output components
    result.update({'W_conv1': [5, 5, 3, 32]})
    W_conv1 = weight_variable([5, 5, 3, 32], "weights_conv1")
    logging.info("First convolutional layer: [5, 5, 3, 32]")
    # a bias vector with a component for each output channel
    b_conv1 = bias_variable([32], "bias_conv1")
    # Apply the layer by convolving x with the weight tensor, add the bias, and apply the ReLU function
    h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
    # Finally max pool with a 2x2 patch. The result has shape (?, 62, 8, 32)
    h_pool1 = max_pool_2x2(h_conv1)

    # Second Convolutional Layer.
    # Stacks a second layer that provides 64 features for each 5x5 patch
    result.update({'W_conv2': [5, 5, 32, 64]})
    W_conv2 = weight_variable([5, 5, 32, 64], "weights_conv2")
    b_conv2 = bias_variable([64], "bias_conv2")
    logging.info("Second convolutional layer: [5, 5, 32, 64]")

    # h_conv2 has dimension (?, 62, 16, 64)
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    # h_pool2 has dimension (?, 31, 8, 64)
    h_pool2 = max_pool_2x2(h_conv2)

    # Densely Connected Layer. The image size has been reduced to 31x4. A fully-connected layer with 1024 neurons is
    # added to allow processing on the entire image. The tensor from the pooling layer is reshaped into a batch of
    # vectors, multiplied by a weight matrix, added to a bias, and applied to a ReLU
    result.update({'W_fc1': [31 * 8 * 64, 1024]})
    W_fc1 = weight_variable([31 * 8 * 64, 1024], "weights_fc1")
    logging.info("First densely-condensed layer: [31 * 8 * 64, 1024]")
    b_fc1 = bias_variable([1024], "bias_fc1")

    h_pool2_flat = tf.reshape(h_pool2, [-1, 31 * 8 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # Dropout. TRo reduce overfitting, we will apply dropout before the readout layer. We create a placeholder for the
    # probability that a neuron's output is kept during dropout. This allows us to turn dropout on during training, and
    # turn it off during testing. TensorFlow's tf.nn.dropout op automatically handles scaling neuron outputs in addition
    # to masking them, so dropout just works without any additional scaling.
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # Readout Layer
    # Finally, we add a layer, just like for the one layer softmax regression above.
    result.update({'W_fc2': [1024, 6]})
    logging.info("Readout layer: [1024, 6]")
    W_fc2 = weight_variable([1024, bm.n_classes], "weights_fc2")
    b_fc2 = bias_variable([bm.n_classes], "bias_fc2")

    # implements the convolutional model
    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    sess = tf.Session()
    saver = tf.train.Saver()

    result.update({'input_uid': input_uid, 'batch_size': bm.batch_size})
    with sess.as_default():
        try:
            saver.restore(sess, "/home/claudio/Projects/eeg_vision/scripts/model_%s.ckpt" % db_uid)
            logging.info("Successfully restored model from the DB")
        except Exception as e:
            logging.info("Failed to restore model from the DB: %s", e)
            sys.exit(0)
        test_classification = []
        while bm.next_test():
            test_classification += [(int(np.argmax(bm.labels('test')[n])),
                                     int(sess.run(tf.argmax(y_conv, 1), feed_dict={
                                         x: bm.samples('test')[n, :, :, :][None, :, :, :], keep_prob: 1.0})))
                                    for n in range(bm.samples('test').shape[0])]
        test_accuracy = float(np.mean([int(a == b) for a, b in test_classification]))
        result.update({'test_accuracy': test_accuracy, 'test_classification': json.dumps(test_classification)})
        logging.info("test accuracy %g", test_accuracy)
    return result


if __name__ == '__main__':
    logging.info("Using Deep Learning for Brainwave Classification")
    data_saver = DataSaver()
    db_uid = '9a9d89058dbaa16687ede93d38a051e8'
    doc = train(db_uid)
    # try:
    #     doc_id = data_saver.save(settings.MONGO_DNN_COLLECTION, doc=doc)
    # except Exception, e:
    #     logging.error("FAILED TO SAVE RESULT IN THE DATABASE:\n%s\n%s", e, traceback.format_exc())
    #     sys.exit(1)
    # logging.info("Successfully saved the result in the DB: doc #%s", doc_id)
    logging.info("Complete.")
