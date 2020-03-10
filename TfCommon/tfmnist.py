# Load the MNIST and normalize images
from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf


def normalize(x):
    """Normalize images.
    :param x: original images
    :return: normalized images
    """
    x = x.astype('float32') / 255.0
    shape = x.shape[1:]
    x = x.reshape(-1, shape[0], shape[1], 1)
    return x


def load_mnist():
    """Load the MNIST dataset
    :return: MNIST data
    """
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = normalize(x_train)
    x_test = normalize(x_test)
    return x_train, y_train, x_test, y_test


if __name__ == '__main__':
    pass
