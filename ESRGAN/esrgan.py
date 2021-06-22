from numpy.core.defchararray import array
from utils import normalize
import ipdb
from ESRGAN.metrics import psnr_metric
from ESRGAN.losses import gan_loss, l1_loss, perceptual_loss
from ESRGAN.data_manager import DataManager
from measures.time_measure import time_context
from ESRGAN.discriminator import build_discriminator
from ESRGAN.rrdbnet import build_rrdbnet
import numpy as np

from keras import Input
from keras.models import Model
import tensorflow_addons as tfa
import tensorflow as tf
from keras.optimizers import Adam

# Input shapes
channels = 3

lr_height = 64
lr_width = 64
lr_shape = (lr_height, lr_width, channels)

hr_height = lr_height * 4
hr_width = lr_width * 4
hr_shape = (hr_height, hr_width, channels)

patch = int(hr_height / 2**5)
discriminator_patch = (patch, patch, 1)

dataset_name = 'img_align_celeba'
dataset_dir = '../datasets/{}/'
data_manager = DataManager(dataset_dir, dataset_name, hr_shape, lr_shape)

optimizer = Adam(
    learning_rate=1e-4,
    #  beta_1=0.9,
    #  beta_2=0.999,
    #  amsgrad=True,
    epsilon=1e-08)

# opt = tf.keras.optimizers.RMSprop(learning_rate=0.02, momentum=0.9, epsilon=0.1)
optimizer = tfa.optimizers.MovingAverage(optimizer)

data_manager.initialize_dirs(2)


def build_esrgan_net():
    # Define a rede geradora
    generator_net = build_rrdbnet()
    generator_net.compile(loss=[l1_loss],
                          loss_weights=[1.0],
                          optimizer=optimizer,
                          metrics=psnr_metric)
    # define network net_generator with Exponential Moving Average (EMA)
    # load pretrained model

    # Define a rede discriminadora
    discriminator_net = build_discriminator()
    discriminator_net.compile(loss=gan_loss,
                              optimizer=optimizer,
                              metrics='accuracy')
    # load pretrained models

    # img_hr = Input(shape=hr_shape)
    img_lr = Input(shape=lr_shape)

    # Generate high res. version from low res.
    fake_hr = generator_net(img_lr)

    # For the adversarial model we will only train the generator
    discriminator_net.trainable = False

    # Discriminator determines validity of generated high res. images
    validity = discriminator_net(fake_hr)

    adversarial = Model([img_lr], [validity], name='ESRGAN')
    adversarial.summary()

    adversarial.compile(loss=gan_loss, optimizer=optimizer)

    def train_esrgan(epochs=100, batch_size=1, sample_interval=50):
        gan_losses = {
            'd_loss': [],
            'g_loss': [],
            'd_fake_loss': [],
            'd_real_loss': []
        }
        disc_interval = 15
        disc_init = 50
        with time_context('treino total'):
            with tf.device('/gpu:0') as GPU:
                for epoch in range(epochs):
                    #  Train Discriminator
                    discriminator_net.trainable = True

                    # Sample images and their conditioning counterparts
                    images = data_manager.load_data(batch_size=batch_size)
                    hr_imgs, lr_imgs = data_manager.prepare_data(images)

                    # From low res. image generate high res. version
                    pred_hr = generator_net.predict(lr_imgs)
                    # ipdb.set_trace()

                    valid = np.ones(
                        (batch_size, )) - np.random.random_sample(batch_size)
                    fake = np.zeros(
                        (batch_size,
                         )) - np.random.random_sample(batch_size) * 0.1

                    # Train the discriminators (original images = real / generated = Fake)
                    d_loss_real = discriminator_net.train_on_batch(
                        hr_imgs, valid)
                    d_loss_fake = discriminator_net.train_on_batch(
                        pred_hr, fake)
                    gan_losses['d_real_loss'].append(d_loss_real[0])
                    gan_losses['d_fake_loss'].append(d_loss_fake[0])
                    d_loss = 0.5 * np.add(d_loss_real[0], d_loss_fake[0])
                    gan_losses['d_loss'].append(d_loss)
                    d_loss = np.array([d_loss])

                    if epoch > disc_init and (epoch % disc_interval) == 0:
                        #  Train Generator
                        discriminator_net.trainable = False

                        # Sample images and their conditioning counterparts
                        images = data_manager.load_data(batch_size=batch_size)
                        _, lr_imgs = data_manager.prepare_data(images)

                        # The generators want the discriminators to label the generated images as real
                        if abs(d_loss_fake[0]) > 25:
                            valid = np.ones(
                                (batch_size,
                                 )) - np.random.random_sample(batch_size) * 0.1
                        else:
                            valid = np.zeros(
                                (batch_size,
                                 )) - np.random.random_sample(batch_size) * 0.1
                        # valid = np.ones((batch_size, )) - np.random.random_sample(batch_size)*0.2

                        # Train the generators
                        g_loss = adversarial.train_on_batch(lr_imgs, valid)
                        gan_losses['g_loss'].append(g_loss)

                    # If at save interval => save generated image samples
                    if epoch % sample_interval == 0:
                        data_manager.sample_images(generator_net, epoch, 2)

            return gan_losses

    return train_esrgan