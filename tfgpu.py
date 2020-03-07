# Initialize the GPU
from __future__ import absolute_import, division, print_function, unicode_literals

import os
import tensorflow as tf


def initialize_gpu(gpu_on):
    """Initialize GPU
    :param gpu_on: GPU activation or not
    :return: number of active GPU devices
    """
    n_gpus = 0
    if gpu_on:
        gpu_list = tf.config.list_physical_devices('GPU')
        if gpu_list:
            try:
                # Currently, memory growth needs to be the same across GPUs
                for gpu in gpu_list:
                    tf.config.experimental.set_memory_growth(gpu, True)
                n_gpus = len(gpu_list)
            except RuntimeError as e:
                # Memory growth must be set before GPUs have been initialized
                print(e)
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    return n_gpus


if __name__ == '__main__':
    pass
