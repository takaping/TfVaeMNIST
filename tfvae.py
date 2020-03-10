# Variational Autoencoder
from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer, Conv2D, Flatten, Dense, Reshape, Conv2DTranspose
import numpy as np
import matplotlib.pyplot as plt
import pprint
from tfcommon import tfmnist, tfgpu


class Sampler(Layer):
    """Class of the Reparameterization Trick Layer
    """
    def __init__(self, name='Sampler', **kwargs):
        """Initialize
        :param name: class name
        :param kwargs:
        """
        super(Sampler, self).__init__(name=name, **kwargs)

    def call(self, inputs, **kwargs):
        """Call
        :param inputs: mean and log. variance space of the normal distribution
        :param kwargs:
        :return: latent space
        """
        z_mean, z_logvar = inputs
        epsilon = tf.random.normal(shape=z_mean.shape)
        return epsilon * tf.exp(z_logvar * .5) + z_mean


class Encoder(Layer):
    """Class of the Encoder Layer
    """
    def __init__(self, filters, kernel_size, strides, latent_dim, name='Encoder', **kwargs):
        """Initialize
        :param filters: numbers of output filters in the convolution
        :param kernel_size: length of the convolution window
        :param strides: stride length of the convolution
        :param latent_dim: dimensionality of the latent space
        :param name: class name
        :param kwargs:
        """
        super(Encoder, self).__init__(name=name, **kwargs)
        self.conv1 = Conv2D(filters=filters[0], kernel_size=kernel_size, strides=strides, activation='relu')
        self.conv2 = Conv2D(filters=filters[1], kernel_size=kernel_size, strides=strides, activation='relu')
        self.flat = Flatten()
        self.dens_mean = Dense(units=latent_dim)
        self.dens_logvar = Dense(units=latent_dim)
        self.sampler = Sampler()

    def call(self, inputs, **kwargs):
        """Call
        :param inputs: original space
        :param kwargs:
        :return: mean, log. variance space of the normal distribution and latent space
        """
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.flat(x)
        z_mean = self.dens_mean(x)
        z_logvar = self.dens_logvar(x)
        z = self.sampler((z_mean, z_logvar))
        return z_mean, z_logvar, z


class Decoder(Layer):
    """Class of the Decoder Layer
    """
    def __init__(self, intermediate_shape,
                 filters, kernel_size, strides, sigmoid_on,
                 name='Decoder', **kwargs):
        """Initialize
        :param intermediate_shape: shape of the intermediate space
        :param filters: numbers of output filters in the convolution
        :param kernel_size: length of the convolution window
        :param strides: stride length of the convolution
        :param sigmoid_on: sigmoid computation or not
        :param name: class name
        :param kwargs:
        """
        super(Decoder, self).__init__(name=name, **kwargs)
        units = tf.reduce_prod(intermediate_shape)
        self.sigmoid_on = sigmoid_on
        self.dens = Dense(units=units, activation='relu')
        self.reshape = Reshape(target_shape=intermediate_shape)
        self.conv_t1 = Conv2DTranspose(filters=filters[0], kernel_size=kernel_size, strides=strides, padding='SAME', activation='relu')
        self.conv_t2 = Conv2DTranspose(filters=filters[1], kernel_size=kernel_size, strides=strides, padding='SAME', activation='relu')
        self.conv_t3 = Conv2DTranspose(filters=filters[2], kernel_size=kernel_size, strides=1, padding='SAME')

    def call(self, inputs, **kwargs):
        """Call
        :param inputs: latent space
        :param kwargs:
        :return: reconstructed space
        """
        x = self.dens(inputs)
        x = self.reshape(x)
        x = self.conv_t1(x)
        x = self.conv_t2(x)
        x_recon = self.conv_t3(x)
        if self.sigmoid_on:
            x_recon = tf.sigmoid(x_recon)
        return x_recon


class Vae(Model):
    """Class of the Variational Autoencoder Model
    """
    def __init__(self, filters=(1, 32, 64), kernel_size=3, strides=2,
                 latent_dim=50, intermediate_shape=(7, 7, 32), sigmoid_on=False,
                 name='VAE', **kwargs):
        """Initialize
        :param filters: numbers of output filters in the convolution
        :param kernel_size: length of the convolution window
        :param strides: stride length of the convolution
        :param latent_dim: dimensionality of the latent space
        :param intermediate_shape: shape of the intermediate space
        :param sigmoid_on: sigmoid computation or not
        :param name: class name
        :param kwargs:
        """
        super(Vae, self).__init__(name=name, **kwargs)
        self.encoder = Encoder(filters[1:], kernel_size, strides, latent_dim)
        self.decoder = Decoder(intermediate_shape, filters[::-1], kernel_size, strides, sigmoid_on)

    def call(self, inputs, **kwargs):
        """Call
        :param inputs: original space
        :param kwargs:
        :return: latent space and reconstructed space
        """
        z_mean, z_logvar, z = self.encoder(inputs)
        x_recon = self.decoder(z)
        return z_mean, z_logvar, z, x_recon


def log_normal_pdf(z, z_mean, z_logvar, raxis=1):
    """
    :param z:
    :param z_mean:
    :param z_logvar:
    :param raxis:
    :return:
    """
    log2pi = tf.math.log(2. * np.pi)
    return tf.reduce_sum(-0.5 * ((z - z_mean) ** 2. * tf.exp(-z_logvar) + z_logvar + log2pi),
                         axis=raxis)


@tf.function
def compute_loss(x, z_mean, z_logvar, z, x_recon):
    """Compute loss
    :param x: original space
    :param z_mean: mean space of the normal distribution
    :param z_logvar: log. variance space of the normal distribution
    :param z: latent space
    :param x_recon: reconstructed space
    :return: loss (ELBO)
    """
    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_recon, labels=x)
    log_pxz = -tf.reduce_sum(cross_entropy, axis=[1, 2, 3])
    log_pz = log_normal_pdf(z, 0., 0.)
    log_qzx = log_normal_pdf(z, z_mean, z_logvar)
    return -tf.reduce_mean(log_pxz + log_pz - log_qzx)


@tf.function
def train_step(x, model, optimizer, loss_metrics):
    """Training step
    :param x: original space
    :param model:
    :param optimizer:
    :param loss_metrics:
    """
    with tf.GradientTape() as tape:
        z_mean, z_logvar, z, x_recon = model(x)
        loss_value = compute_loss(x, z_mean, z_logvar, z, x_recon)
    grads = tape.gradient(loss_value, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    loss_metrics(loss_value)


@tf.function
def test_step(x, model, loss_metrics):
    """Testing step
    :param x: original space
    :param model:
    :param loss_metrics:
    :return: reconstructed space
    """
    z_mean, z_logvar, z, x_recon = model(x)
    loss_value = compute_loss(x, z_mean, z_logvar, z, x_recon)
    loss_metrics(loss_value)
    return x_recon


def delete_image(x_data, y_data, delete_num):
    new_data = []
    append_x = new_data.append
    for x, y in zip(x_data, y_data):
        if delete_num != y:
            append_x(x)
    return np.array(new_data)


def perform_training(x_data, y_data, model, optimizer, delete_num=5, batch_size=64, epochs=10):
    """Perform training
    :param x_data: original space
    :param y_data: label space
    :param model:
    :param optimizer:
    :param batch_size: batch size
    :param epochs: epoch number
    """
    x_data = delete_image(x_data, y_data, delete_num)
    dataset = tf.data.Dataset.from_tensor_slices((x_data)).shuffle(x_data.shape[0]).batch(batch_size)
    loss_metrics = tf.keras.metrics.Mean(name='train_loss')
    template = 'Epoch %s: Loss = %s'
    for epoch in range(epochs):
        for x in dataset:
            train_step(x, model, optimizer, loss_metrics)
        print(template % (epoch + 1,
                          -loss_metrics.result()))
        loss_metrics.reset_states()


def classify_image(x_data, y_data):
    x_list = [[] for i in range(10)]
    for x, y in zip(x_data, y_data):
        x_list[y].append(x)
    return [np.array(x_list[i]) for i in range(10)]


def perform_testing(x_data, y_data, model, batch_size=64):
    """Perform testing
    :param x_data: original space
    :param y_data: label space
    :param model:
    :param batch_size: batch size
    """
    x_list = classify_image(x_data, y_data)
    loss_metrics = tf.keras.metrics.Mean(name='test_loss')
    template = 'Num. %s: Loss = %s'
    res_list = []
    append_result = res_list.append
    for step, x_data in enumerate(x_list):
        dataset = tf.data.Dataset.from_tensor_slices((x_data)).batch(batch_size)
        for x in dataset:
            test_step(x, model, loss_metrics)
        append_result(template % (step, -loss_metrics.result()))
        loss_metrics.reset_states()
    pprint.pprint(res_list)



def perform_prediction(x_data, y_data, model, batch_size=1):
    """Perform prediction
    :param x_data: original space
    :param y_data: label space
    :param model:
    :param batch_size: batch size
    """
    dataset = tf.data.Dataset.from_tensor_slices((x_data, y_data)).batch(batch_size)
    loss_metrics = tf.keras.metrics.Mean(name='pred_loss')
    shape = x_data.shape[1:3]
    image_title = 'Label = %d'
    recon_title = 'Loss = %.2f'
    template = 'Number %s: Loss = %s'
    for x, y in dataset:
        x_recon = test_step(x, model, loss_metrics)
        loss = -loss_metrics.result()
        print(template % (y, loss))

        plt.subplot(1, 2, 1)
        plt.axis('off')
        plt.title(image_title % (y,))
        plt.imshow(np.array(x).reshape(shape), cmap='gray')
        plt.subplot(1, 2, 2)
        plt.title(recon_title % (loss,))
        plt.imshow(np.array(x_recon).reshape(shape), cmap='gray')
        plt.show()
        loss_metrics.reset_states()


if __name__ == '__main__':
    # GPU activation or not.
    gpu_on = True
    # processing number
    #   0: training
    #   1: testing
    #   the others: prediction
    proc_num = 0

    tfgpu.initialize_gpu(gpu_on)    # Initialize GPU devices

    x_train, y_train, x_test, y_test = tfmnist.load_mnist()     # Load the MNIST dataset

    # Instantiate the model
    model = Vae()
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    # Perform training, testing or prediction
    if proc_num == 0:       # training
        perform_training(x_train, y_train, model, optimizer, delete_num=5, epochs=10)
        model.save_weights('model', save_format='tf')
    elif proc_num == 1:     # testing
        model.load_weights('model')
        perform_testing(x_test, y_test, model)
    else:                   # prediction
        model.load_weights('model')
        perform_prediction(x_test, y_test, model)
