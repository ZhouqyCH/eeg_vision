import functools
import tensorflow as tf


def doublewrap(function):
    """
    A decorator decorator, allowing to use the decorator to be used without
    parentheses if not arguments are provided. All arguments must be optional.
    """

    @functools.wraps(function)
    def decorator(*args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
            return function(args[0])
        else:
            return lambda wrapee: function(wrapee, *args, **kwargs)

    return decorator


@doublewrap
def define_scope(function, scope=None, *args, **kwargs):
    """
    A decorator for functions that define TensorFlow operations. The wrapped
    function will only be executed once. Subsequent calls to it will directly
    return the result so that operations are added to the graph only once.
    The operations added by the function live within a tf.variable_scope(). If
    this decorator is used with arguments, they will be forwarded to the
    variable scope. The scope name defaults to the name of the wrapped
    function.
    """
    attribute = '_cache_' + function.__name__
    name = scope or function.__name__

    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            with tf.variable_scope(name, *args, **kwargs):
                setattr(self, attribute, function(self))
        return getattr(self, attribute)

    return decorator


class BaseDNN(object):
    def __init__(self, n_channels, trial_size, n_classes, max_iter=20000, learning_rate=1e-3, batch_size=128,
                 display_step=10, dropout=0.5):
        self.max_iter = max_iter
        self.n_channels = n_channels
        self.trial_size = trial_size
        self.n_comps = 3
        self.n_classes = n_classes
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.display_step = display_step
        self.dropout = dropout  # Dropout, probability to keep units
        self.image = None
        self.label = None

    @property
    def outputs(self):
        return []

    @property
    def weights(self):
        return {}

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

    @define_scope
    def cross_entropy(self):
        return tf.nn.softmax_cross_entropy_with_logits(self.prediction, self.label)

    @define_scope
    def cost(self):
        return tf.reduce_mean(self.cross_entropy)

    @define_scope
    def optimize(self):
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        return optimizer.minimize(self.cost)

    @define_scope
    def accuracy(self):
        correct_predictions = tf.equal(tf.argmax(self.label, 1), tf.argmax(self.prediction, 1))
        return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

    @define_scope(initializer=tf.contrib.slim.xavier_initializer())
    def prediction(self):
        weights = self.weights
        biases = self.biases

        # Convolution Layer
        conv1 = self.conv2d(self.image, weights['wc1'], biases['bc1'])
        # Max Pooling (down-sampling)
        conv1_maxpool = self.maxpool2d(conv1, k=2)

        # Convolution Layer
        conv2 = self.conv2d(conv1_maxpool, weights['wc2'], biases['bc2'])
        # Max Pooling (down-sampling)
        conv2_maxpool = self.maxpool2d(conv2, k=2)
        s = conv2_maxpool.get_shape().as_list()

        # Reshape conv2 output to fit the fully connected layer input
        conv2_maxpool_reshaped = tf.reshape(conv2_maxpool, [-1, s[1] * s[2] * s[3]])

        # Fully connected layer
        fc1 = tf.add(tf.matmul(conv2_maxpool_reshaped, weights['wd1']), biases['bd1'])
        fc1 = tf.nn.relu(fc1)

        # Apply Dropout
        fc1_dropout = tf.nn.dropout(fc1, self.dropout)

        # Output, class prediction
        out = tf.add(tf.matmul(fc1_dropout, weights['out']), biases['out'])
        return out

    def fit(self, dataset):
        # tf Graph input
        self.image = tf.placeholder(tf.float32, shape=[None, self.n_channels, self.trial_size, self.n_comps])
        self.label = tf.placeholder(tf.float32, shape=[None, self.n_classes])
        self.optimize
        self.accuracy
        self.prediction
        keep_prob = tf.placeholder(tf.float32)  # dropout (keep probability)

        # Launch the graph
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            step = 1
            # Keep training until reach max iterations
            while step * self.batch_size < self.max_iter:
                batch_x, batch_y = dataset.train.next_batch(self.batch_size)
                # Run optimization op (backprop)
                sess.run(self.optimize, feed_dict={self.image: batch_x, self.label: batch_y, keep_prob: 0.5})
                if step % self.display_step == 0:
                    # Calculate batch loss and accuracy
                    loss, acc = sess.run([self.cost, self.accuracy],
                                         feed_dict={self.image: batch_x, self.label: batch_y, keep_prob: 1.})
                    print("Iter " + str(step * self.batch_size) + ", Minibatch Loss= " + "{:.6f}".format(
                        loss) + ", Training Accuracy= " + "{:.5f}".format(acc))
                step += 1
            print("Optimization Finished!")

            # Calculate accuracy for 256 mnist test images
            print("Testing Accuracy:", sess.run(self.accuracy, feed_dict={self.image: dataset.test.samples,
                                                                          self.label: dataset.test.labels,
                                                                          keep_prob: 1.}))
