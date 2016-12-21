import tensorflow as tf


class BaseDNN(object):
    def __init__(self, n_channels, trial_size, n_classes, max_iter=20000, learning_rate=1e-3, batch_size=128,
                 display_step=10, dropout=0.75):
        self.max_iter = max_iter
        self.n_channels = n_channels
        self.trial_size = trial_size
        self.n_comps = 3
        self.n_classes = n_classes
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.display_step = display_step
        self.dropout = dropout  # Dropout, probability to keep units

    @property
    def strides(self):
        return [[5, 5], [5, 5]]

    @property
    def outputs(self):
        return [16, 32, 64]

    @property
    def weights(self):
        return {
            'wc1': tf.Variable(tf.random_normal(self.strides[0] + [self.n_comps, self.outputs[0]])),
            'wc2': tf.Variable(tf.random_normal(self.strides[1] + [self.outputs[0], self.outputs[1]])),
            'wd1': tf.Variable(tf.random_normal([7 * 7 * self.outputs[1], self.outputs[2]])),
            'out': tf.Variable(tf.random_normal([self.outputs[2], self.n_classes]))
        }

    @property
    def biases(self):
        return {
            'bc1': tf.Variable(tf.random_normal([self.outputs[0]])),
            'bc2': tf.Variable(tf.random_normal([self.outputs[1]])),
            'bd1': tf.Variable(tf.random_normal([self.outputs[2]])),
            'out': tf.Variable(tf.random_normal([self.n_classes]))
        }

    @staticmethod
    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    @staticmethod
    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    @staticmethod
    def conv2d(x, W, b, strides=1):
        # Conv2D wrapper, with bias and relu activation
        x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
        x = tf.nn.bias_add(x, b)
        return tf.nn.relu(x)

    @staticmethod
    def maxpool2d(x, k=2):
        # MaxPool2D wrapper
        return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

    def conv_net(self, x):
        weights = self.weights
        biases = self.biases

        # Convolution Layer
        conv1 = self.conv2d(x, weights['wc1'], biases['bc1'])
        # Max Pooling (down-sampling)
        conv1 = self.maxpool2d(conv1, k=2)

        # Convolution Layer
        conv2 = self.conv2d(conv1, weights['wc2'], biases['bc2'])
        # Max Pooling (down-sampling)
        conv2 = self.maxpool2d(conv2, k=2)

        # Fully connected layer
        # Reshape conv2 output to fit fully connected layer input
        fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
        fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
        fc1 = tf.nn.relu(fc1)
        # Apply Dropout
        fc1 = tf.nn.dropout(fc1, self.dropout)

        # Output, class prediction
        out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
        return out

    def fit(self, dataset):
        # tf Graph input
        x = tf.placeholder(tf.float32, shape=[None, self.n_channels, self.trial_size, self.n_comps])
        y = tf.placeholder(tf.float32, shape=[None, self.n_classes])
        keep_prob = tf.placeholder(tf.float32)  # dropout (keep probability)

        pred = self.conv_net(x)

        # Define loss and optimizer
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(cost)

        # Evaluate model
        correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        # Initializing the variables
        init = tf.initialize_all_variables()

        # Launch the graph
        with tf.Session() as sess:
            sess.run(init)
            step = 1
            # Keep training until reach max iterations
            while step * self.batch_size < self.max_iter:
                batch_x, batch_y = dataset.train.next_batch(self.batch_size)
                # Run optimization op (backprop)
                sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, keep_prob: self.dropout})
                if step % self.display_step == 0:
                    # Calculate batch loss and accuracy
                    loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x, y: batch_y, keep_prob: 1.})
                    print("Iter " + str(step * self.batch_size) + ", Minibatch Loss= " + "{:.6f}".format(
                        loss) + ", Training Accuracy= " + "{:.5f}".format(acc))
                step += 1
            print("Optimization Finished!")

            # Calculate accuracy for 256 mnist test images
            print("Testing Accuracy:", sess.run(accuracy, feed_dict={x: dataset.test.images[:256],
                                                y: dataset.test.labels[:256],
                                                keep_prob: 1.}))
