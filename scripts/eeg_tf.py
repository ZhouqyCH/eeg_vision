import argparse
import os

import tensorflow as tf

import settings
from etc.dataset import build_data_sets


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
    parser = argparse.ArgumentParser()
    parser.add_argument("subject", choices=settings.SUBJECTS)
    args = parser.parse_args()

    file_name = os.path.join(settings.PATH_TO_MAT_FILES, args.subject.upper() + ".mat")
    ds = build_data_sets(file_name, avg_group_size=5, derivation='electric_field', random_state=42, test_proportion=0.2)

    # The input x will consist of a tensor of floating point numbers
    x = tf.placeholder(tf.float32, shape=[None, ds.train.n_channels, ds.train.trial_size, ds.train.n_comps])
    # The target output classes y_ will consist of a 2d tensor, where each row is a one-hot 6-dimensional vector
    # indicating which digit class (zero through 5) the corresponding trial belongs to
    y_ = tf.placeholder(tf.float32, shape=[None, ds.train.n_class])

    # First Convolutional Layer.
    # It will compute 32 features for each 5x5 patch. Its weight tensor will have a shape of
    # [5, 5, 3, 32]. The first two dimensions are the patch size, the next is the number of electric field components,
    # and the last is the number of output components
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

    # Densely Connected Layer. The image size has been reduced to 7x7. A fully-connected layer with 1024 neurons is
    # added to allow processing on the entire image. The tensor from the pooling layer is reshaped into a batch of
    # vectors, multiplied by a weight matrix, added to a bias, and applied to a ReLU
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # Dropout. To reduce overfitting, we will apply dropout before the readout layer. We create a placeholder for the
    # probability that a neuron's output is kept during dropout. This allows us to turn dropout on during training, and
    # turn it off during testing. TensorFlow's tf.nn.dropout op automatically handles scaling neuron outputs in addition
    # to masking them, so dropout just works without any additional scaling.
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # Readout Layer
    # Finally, we add a layer, just like for the one layer softmax regression above.
    W_fc2 = weight_variable([1024, ds.train.n_class])
    b_fc2 = bias_variable([ds.train.n_class])

    # implements the convolutional model
    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    # the loss function is the cross-entropy between the target and the softmax activation function applied to the
    # model's prediction. The function tf.nn.softmax_cross_entropy_with_logits internally applies the softmax on the
    # model's unnormalized model prediction and sums across all classes, and tf.reduce_mean takes the average over
    # these sums.
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))

    # uses steepest gradient descent, with a step length of 0.5, to descend the cross entropy.
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    sess = tf.Session()
    sess.run(tf.initialize_all_variables())

    with sess.as_default():
        # Train the model by repeatedly running train_step.
        max_iter = 20000
        print "Training the network with a maximum of %s iterations" % max_iter
        for i in range(max_iter):
            batch = ds.train.next_batch(50)
            if i % 100 == 0:
                train_accuracy = accuracy.eval(feed_dict={
                    x: batch[0], y_: batch[1], keep_prob: 1.0})
                print("step %d, training accuracy %g" % (i, train_accuracy))

            train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

        print("test accuracy %g" % accuracy.eval(feed_dict={
            x: ds.test.samples, y_: ds.test.labels, keep_prob: 1.0}))

    print "Complete."
