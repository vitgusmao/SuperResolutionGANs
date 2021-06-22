import ipdb
import numpy as np
import tensorflow as tf

from keras import Input
from keras.models import Model

from gans.esrgan.discriminator import build_discriminator
from gans.esrgan.metrics import psnr_metric
from gans.esrgan.rrdbnet import build_rrdbnet
from data_manager import DataManager
from losses import gan_loss, l1_loss, build_perceptual_vgg
from measures.time_measure import time_context
from optimizers import get_adam_optimizer

# Input shapes
channels = 3

lr_height = 64
lr_width = 64
lr_shape = (lr_height, lr_width, channels)

hr_height = lr_height * 4
hr_width = lr_width * 4
hr_shape = (hr_height, hr_width, channels)

dataset_name = 'img_align_celeba'
dataset_dir = '../datasets/{}/'
data_manager = DataManager(dataset_dir, dataset_name, hr_shape, lr_shape)
data_manager.initialize_dirs(2)

optimizer = get_adam_optimizer(lr=2e-4,
                               beta_1=0.5,
                               epsilon=1e-08,
                               moving_avarage=True)

perceptual_loss = build_perceptual_vgg(hr_shape)


def build_esrgan_net():
    # Define a rede geradora
    generator_net = build_rrdbnet()
    generator_net.compile(loss=[l1_loss, perceptual_loss],
                          loss_weights=[0.3, 0.7],
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

    def train_esrgan(epochs=100,
                     batch_size=1,
                     sample_interval=50,
                     initial_values=[1],
                     complement_value=[0]):
        informations = {
            'd_loss': [],
            'g_loss': [],
            'd_fake_loss': [],
            'd_real_loss': [],
            'valid_g': [],
        }

        step_value = [0 for i in range(len(initial_values))]
        divided = int(epochs / len(initial_values))
        init_discriminator = 10

        with time_context('treino total'):
            with tf.device('/gpu:0') as GPU:
                for epoch in range(epochs):
                    #  Train Discriminator
                    discriminator_net.trainable = True

                    # Sample images and their conditioning counterparts
                    hr_imgs, lr_imgs = data_manager.load_prepared_data(
                        batch_size=batch_size)

                    # From low res. image generate high res. version
                    pred_hr = generator_net.predict(lr_imgs)

                    valid = np.ones(
                        (batch_size,
                         )) - np.random.random_sample(batch_size) * 0.1
                    fake = np.zeros((batch_size, ))

                    # Train the discriminators (original images = real / generated = Fake)
                    d_loss_real = discriminator_net.train_on_batch(
                        hr_imgs, valid)
                    d_loss_fake = discriminator_net.train_on_batch(
                        pred_hr, fake)
                    informations['d_real_loss'].append(d_loss_real[0])
                    informations['d_fake_loss'].append(d_loss_fake[0])
                    d_loss = 0.5 * np.add(d_loss_real[0], d_loss_fake[0])
                    informations['d_loss'].append(d_loss)

                    #  Train Generator
                    if epoch > init_discriminator:
                        discriminator_net.trainable = False

                        # Sample images and their conditioning counterparts
                        _, lr_imgs = data_manager.load_prepared_data(
                            batch_size=batch_size)

                        # fake_imgs = generator_net(lr_imgs, training=False).numpy()
                        # mse = tf.keras.losses.mean_squared_error(
                        #     hr_imgs, fake_imgs)
                        # valid = abs(np.sin((epochs + 1)))
                        # valid = np.array([valid])

                        valid_index = int(epoch / divided)
                        step_value[valid_index] += complement_value[
                            valid_index] / divided

                        valid = np.zeros(
                            (batch_size,
                             )) + initial_values[valid_index] + step_value[
                                 valid_index] - np.random.random_sample(
                                     batch_size) * 0.15

                        informations['valid_g'].append(valid[0])

                        print('Epoch: ', epoch)
                        print('ValidIndex: ', valid_index)
                        print('StepValue: ', step_value[valid_index])
                        print('Valid: ', valid)
                        print()

                        # Train the generators
                        g_loss = adversarial.train_on_batch(lr_imgs, valid)
                        informations['g_loss'].append(g_loss)

                    # If at save interval => save generated image samples
                    if epoch % sample_interval == 0:
                        data_manager.sample_images(generator_net, epoch, 2)

            return informations

    return train_esrgan
