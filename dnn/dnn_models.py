import tensorflow as tf

from dnn.base_dnn import BaseDNN


class DNN1(BaseDNN):
    @property
    def outputs(self):
        return [32, 64, 1024]

    @property
    def weights(self):
        return {
            'wc1': tf.Variable(tf.random_normal([5, 5, self.n_comps, self.outputs[0]])),
            'wc2': tf.Variable(tf.random_normal([5, 5, self.outputs[0], self.outputs[1]])),
            'wd1': tf.Variable(tf.random_normal([31 * 8 * self.outputs[1], self.outputs[2]])),
            'out': tf.Variable(tf.random_normal([self.outputs[2], self.n_classes]))
        }
