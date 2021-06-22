from vgg_net import build_vgg
import numpy as np
import tensorflow as tf

from keras import backend as K
from keras import Input
from keras.models import Model

from data_manager import DataManager
from measures.time_measure import time_context
from optimizers import get_adam_optimizer
from gans.srgan.discriminator import build_discriminator
from gans.srgan.generator import build_generator
from losses import l1_loss

# Input shape
channels = 3
lr_height = 64
lr_width = 64
lr_shape = (lr_height, lr_width, channels)
hr_height = lr_height * 4
hr_width = lr_width * 4
hr_shape = (hr_height, hr_width, channels)

dis_patch = (8, 8, 1)

# Number of residual blocks in the generator
n_residual_blocks = 16

optimizer = get_adam_optimizer(lr=1e-4, beta_1=0.5, epsilon=1e-08)

dataset_name = 'img_align_celeba'
dataset_dir = '../datasets/{}/'

data_manager = DataManager(dataset_dir, dataset_name, hr_shape, lr_shape)
data_manager.initialize_dirs(2)


def build_srgan_net():

    vgg = build_vgg(hr_shape, tl_layer='block2_conv2')
    # vgg.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])

    # Build the generator
    generator = build_generator(lr_shape)
    generator.compile(loss=[l1_loss], optimizer=optimizer)

    # Build and compile the discriminator
    discriminator = build_discriminator(hr_shape)
    discriminator.compile(loss='binary_crossentropy',
                          optimizer=optimizer,
                          metrics=['accuracy'])

    # High res. and low res. images
    img_hr = Input(shape=hr_shape)
    img_lr = Input(shape=lr_shape)

    # Generate high res. version from low res.
    fake_hr = generator(img_lr)

    # Extract image features of the generated img
    fake_features = vgg(fake_hr)

    # For the adversarial model we will only train the generator
    discriminator.trainable = False

    # Discriminator determines validity of generated high res. images
    validity = discriminator(fake_hr)

    adversarial = Model([img_lr, img_hr], [validity, fake_features])
    adversarial.compile(loss=['binary_crossentropy'],
                        loss_weights=[1e-3, 1],
                        optimizer=optimizer)

    def train_srgan(
        epochs=100,
        batch_size=1,
        sample_interval=50,
    ):
        informations = {
            'd_loss': [],
            'g_loss1': [],
            'g_loss2': [],
            'd_fake_loss': [],
            'd_real_loss': [],
            'valid_g': [],
        }

        with time_context('treino total'):
            with tf.device('/gpu:0') as GPU:
                for epoch in range(epochs):
                    # ----------------------
                    #  Train Discriminator
                    # ----------------------

                    # Sample images and their conditioning counterparts
                    imgs_hr, imgs_lr = data_manager.load_prepared_data(
                        batch_size)

                    # From low res. image generate high res. version
                    fake_hr = generator.predict(imgs_lr)

                    valid = np.ones((batch_size, ) + dis_patch) * 0.1
                    fake = np.zeros((batch_size, ) + dis_patch)

                    discriminator.trainable = True

                    # Train the discriminators (original images = real / generated = Fake)
                    d_loss_real = discriminator.train_on_batch(imgs_hr, valid)
                    d_loss_fake = discriminator.train_on_batch(fake_hr, fake)
                    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
                    informations['d_real_loss'].append(d_loss_real[0])
                    informations['d_fake_loss'].append(d_loss_fake[0])
                    informations['d_loss'].append(d_loss[0])

                    # ------------------
                    #  Train Generator
                    # ------------------

                    # Sample images and their conditioning counterparts
                    imgs_hr, imgs_lr = data_manager.load_prepared_data(
                        batch_size)

                    discriminator.trainable = False

                    # The generators want the discriminators to label the generated images as real
                    valid = np.ones((batch_size, ) + dis_patch)

                    # Extract ground truth image features using pre-trained VGG19 model
                    image_features = vgg.predict(imgs_hr)

                    # Train the generators
                    g_loss = adversarial.train_on_batch(
                        [imgs_lr, imgs_hr], [valid, image_features])
                    informations['g_loss1'].append(g_loss[0])
                    informations['g_loss2'].append(g_loss[0])

                    # If at save interval => save generated image samples
                    if epoch % sample_interval == 0:
                        data_manager.sample_images(generator, epoch, 2)

        return informations

    return train_srgan