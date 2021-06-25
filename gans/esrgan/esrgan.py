import ipdb
import numpy as np
import os
import tensorflow as tf

from keras import Input
from keras.models import Model

from data_manager import DataManager
from gans.esrgan.discriminator import build_discriminator
from gans.esrgan.metrics import psnr_metric
from gans.esrgan.rrdbnet import build_rrdbnet
from losses import build_vgg_loss
from measures.time_measure import time_context
from optimizers import get_adam_optimizer
from vgg_net import build_vgg

# Input shapes
channels = 3

lr_height = 64
lr_width = 64
lr_shape = (lr_height, lr_width, channels)

hr_height = lr_height * 4
hr_width = lr_width * 4
hr_shape = (hr_height, hr_width, channels)

dis_patch = (8, 8, 1)

dataset_name = 'img_align_celeba'
dataset_dir = '../datasets/{}/'
models_path = 'models/esrgan/{}'
data_manager = DataManager(dataset_dir, dataset_name, hr_shape, lr_shape)


def build_esrgan_net():

    optimizer = get_adam_optimizer(lr=2e-4,
                                   beta_1=0.5,
                                   epsilon=1e-08,
                                   moving_avarage=True)

    vgg = build_vgg(hr_shape, full_net=True)

    vgg_loss = build_vgg_loss(vgg)

    # Define a rede geradora
    if os.path.isfile(models_path.format('generator_net')):
        generator_net = tf.keras.models.load_model(
            models_path.format('generator_net'))
    else:
        #     os.makedirs(models_path.format('generator_net'), exist_ok=True)
        generator_net = build_rrdbnet()
        generator_net.compile(
            loss=['mse', 'mae'],
            #   loss_weights=[0.3, 0.7],
            optimizer=optimizer,
            metrics=psnr_metric)

    # Define a rede discriminadora
    if os.path.isfile(models_path.format('discriminator_net')):
        discriminator_net = tf.keras.models.load_model(
            models_path.format('discriminator_net'))
    else:
        discriminator_net = build_discriminator()
        discriminator_net.compile(loss=['binary_crossentropy'],
                                  optimizer=optimizer,
                                  metrics='accuracy')

    if os.path.isfile(models_path.format('adversarial_net')):
        adversarial = tf.keras.models.load_model(
            models_path.format('adversarial_net'))
    else:
        # For the adversarial model we will only train the generator
        discriminator_net.trainable = False

        img_input = Input(shape=lr_shape)

        # Generate high res. version from low res.
        gen_hr = generator_net(img_input)

        # Discriminator determines validity of generated high res. images
        validity = discriminator_net(gen_hr)

        adversarial = Model([img_input], [gen_hr, validity], name='ESRGAN')
        adversarial.summary()

        adversarial.compile(loss=[vgg_loss], optimizer=optimizer)

    def train_esrgan(
        epochs=100,
        batch_size=1,
        sample_interval=50,
    ):
        data_manager.initialize_dirs(2, epochs)
        informations = {
            'd_loss': [],
            'g_loss': [],
            'd_fake_loss': [],
            'd_real_loss': [],
        }

        with time_context('treino total'):
            with tf.device('/gpu:0'):
                for epoch in range(epochs):
                    #  Train Discriminator
                    discriminator_net.trainable = True

                    # Sample images and their conditioning counterparts
                    hr_imgs, lr_imgs = data_manager.load_prepared_data(
                        batch_size=batch_size)

                    # From low res. image generate high res. version
                    pred_hr = generator_net.predict(lr_imgs)

                    real_y = np.ones(
                        (batch_size, ) +
                        dis_patch) - np.random.random_sample(batch_size) * 0.1
                    fake_y = np.zeros((batch_size, ) + dis_patch)

                    # Train the discriminators (original images = real / generated = Fake)
                    d_loss_real = discriminator_net.train_on_batch(
                        hr_imgs, real_y)
                    d_loss_fake = discriminator_net.train_on_batch(
                        pred_hr, fake_y)
                    informations['d_real_loss'].append(d_loss_real[0])
                    informations['d_fake_loss'].append(d_loss_fake[0])
                    d_loss = 0.5 * np.add(d_loss_real[0], d_loss_fake[0])
                    informations['d_loss'].append(d_loss)

                    #  Train Generator
                    discriminator_net.trainable = False

                    # Sample images and their conditioning counterparts
                    hr_imgs, lr_imgs = data_manager.load_prepared_data(
                        batch_size=batch_size)

                    vgg_y = vgg.predict(hr_imgs)

                    # Train the generators
                    g_loss = adversarial.train_on_batch(
                        lr_imgs, [hr_imgs, vgg_y])
                    informations['g_loss'].append(g_loss[0])

                    if epoch > 0:
                        if informations['d_loss'][
                                epoch - 1] >= informations['d_loss'][epoch]:
                            discriminator_net.save(
                                models_path.format('discriminator_net'))

                        if informations['g_loss'][
                                epoch - 1] >= informations['g_loss'][epoch]:
                            adversarial.save(
                                models_path.format('adversarial_net'))
                            generator_net.save(
                                models_path.format('generator_net'))

                    if epoch % sample_interval == 0:
                        data_manager.sample_images(generator_net, epoch, 2)

        return informations

    return train_esrgan
