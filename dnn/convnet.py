import logging
import traceback
from datetime import datetime

import numpy as np
import tensorflow as tf

from data_tools.utils import OneHotEncoder


class ConvNet(object):
    def __init__(self, W_conv1, bias_conv1, W_conv2, bias_conv2, W_fc1, bias_fc1, W_fc2, bias_fc2, trial_size,
                 n_comps, n_channels, n_classes, learning_rate=1e-3):
        self.W_conv1 = W_conv1
        self.bias_conv1 = bias_conv1
        self.W_conv2 = W_conv2
        self.bias_conv2 = bias_conv2
        self.W_fc1 = W_fc1
        self.bias_fc1 = bias_fc1
        self.W_fc2 = W_fc2
        self.bias_fc2 = bias_fc2
        self.trial_size = trial_size
        self.n_comps = n_comps
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.learning_rate = learning_rate
        self.label_encoder = OneHotEncoder().fit(range(1, n_classes+1))
        self.train_accuracy = list()

    def train(self, batcher, output_filename=None):
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

        # The input x will consist of a tensor of floating point numbers of shape (?, 124, 32, 3)
        x = tf.placeholder(tf.float32, shape=[None, self.n_channels, self.trial_size, self.n_comps])
        # The target output classes y_ will consist of a 2d tensor, where each row is a one-hot 6-dimensional vector
        # indicating which digit class (zero through 5) the corresponding trial belongs to
        y_ = tf.placeholder(tf.float32, shape=[None, self.n_classes])

        # First Convolutional Layer.
        W_conv1 = weight_variable(self.W_conv1, "W_conv1")
        # a bias vector with a component for each output channel
        b_conv1 = bias_variable(self.bias_conv1, "bias_conv1")
        # Apply the layer by convolving x with the weight tensor, add the bias, and apply the ReLU function
        h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
        # Finally max pool with a 2x2 patch. The result has shape (?, 62, 8, 32)
        h_pool1 = max_pool_2x2(h_conv1)

        # Second Convolutional Layer.
        W_conv2 = weight_variable(self.W_conv2, "W_conv2")
        b_conv2 = bias_variable(self.bias_conv2, "bias_conv2")
        # h_conv2 has dimension (?, 62, 16, 64)
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
        # h_pool2 has dimension (?, 31, 8, 64)
        h_pool2 = max_pool_2x2(h_conv2)

        # Densely-connected Layer.
        W_fc1 = weight_variable(self.W_fc1, "weights_fc1")
        b_fc1 = bias_variable(self.bias_fc1, "bias_fc1")
        h_pool2_flat = tf.reshape(h_pool2, [-1, self.W_fc1[0]])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

        # Dropout. TRo reduce overfitting, we will apply dropout before the readout layer. We create a placeholder for
        # the probability that a neuron's output is kept during dropout. This allows us to turn dropout on during
        # training, and turn it off during testing. TensorFlow's tf.nn.dropout op automatically handles scaling neuron
        # outputs in addition to masking them, so dropout just works without any additional scaling.
        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

        # Readout Layer
        # Finally, we add a layer, just like for the one layer softmax regression above.
        W_fc2 = weight_variable(self.W_fc2, "weights_fc2")
        b_fc2 = bias_variable(self.bias_fc2, "bias_fc2")

        # implements the convolutional model
        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

        # the loss function is the cross-entropy between the target and the softmax activation function applied to the
        # model's prediction. The function tf.nn.softmax_cross_entropy_with_logits internally applies the softmax on the
        # model's unnormalized model prediction and sums across all classes, and tf.reduce_mean takes the average over
        # these sums.
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))

        # uses steepest gradient descent to descend the cross entropy.
        train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(cross_entropy)

        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        saver = tf.train.Saver()
        sess = tf.Session()
        sess.run(tf.initialize_all_variables())

        self.train_accuracy = []
        with sess.as_default():
            logging.info("Training the network with a maximum of %s batches of size %s", batcher.count_max,
                         batcher.size)
            last_iter = 0
            samples, labels = batcher.next_batch()
            labels = np.asarray([self.label_encoder.transform(lab) for lab in labels])
            while samples is not None:
                acc = accuracy.eval(feed_dict={x: samples, y_: labels, keep_prob: 1.0})
                logging.info("%s: last iter %d - training accuracy: %g", datetime.now().isoformat(), last_iter, acc)
                train_step.run(feed_dict={x: samples, y_: labels, keep_prob: 0.5})
                last_iter += batcher.size
                self.train_accuracy.append({'last_iter': last_iter, 'acc': float(acc)})
            if output_filename:
                try:
                    save_path = saver.save(sess, output_filename)
                    logging.info("Successfully saved the model to %s", save_path)
                except Exception as e:
                    logging.error("Failed to save the model: %s\n%s", e, traceback.format_exc())
        return self.train_accuracy
