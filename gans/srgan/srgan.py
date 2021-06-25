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

optimizer = get_adam_optimizer(lr=1e-4, amsgrad=True, epsilon=1e-08)

dataset_name = 'img_align_celeba'
dataset_dir = '../datasets/{}/'

data_manager = DataManager(dataset_dir, dataset_name, hr_shape, lr_shape)

vgg = build_vgg(hr_shape, full_net=True)


def vgg_loss(y_true, y_pred):
    return K.mean(K.square(vgg(y_true) - vgg(y_pred)))


def build_srgan_net():
    # Build the generator
    generator = build_generator(lr_shape)
    generator.compile(loss=['mse', 'mae'], optimizer=optimizer)

    discriminator = build_discriminator(hr_shape)
    discriminator.compile(loss=['binary_crossentropy'],
                          optimizer=optimizer,
                          metrics=['accuracy'])

    discriminator.trainable = False

    img_input = Input(shape=lr_shape)

    gen_hr = generator(img_input)

    validity_output = discriminator(gen_hr)

    adversarial = Model(inputs=img_input, outputs=[gen_hr, validity_output])
    adversarial.compile(loss=[vgg_loss],
                        loss_weights=[1.0, 1e-3],
                        optimizer=optimizer)

    # imgs_hr, imgs_lr = data_manager.load_prepared_data(1)
    # x = discriminator.predict(imgs_hr)
    # y = generator.predict(imgs_lr)
    # z = adversarial.predict(imgs_lr)
    # w = vgg.predict(imgs_hr)
    # import ipdb; ipdb.set_trace()

    def train_srgan(
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
            'g_loss1': [],
            'g_loss2': [],
        }
        with time_context('treino total'):
            with tf.device('/gpu:0') as GPU:
                for epoch in range(epochs):
                    # ----------------------
                    #  Train Discriminator

                    imgs_hr, imgs_lr = data_manager.load_prepared_data(
                        batch_size)

                    fake_hr = generator.predict(imgs_lr)

                    real_y = np.ones(
                        (batch_size, ) +
                        dis_patch) - np.random.random_sample(batch_size) * 0.1
                    fake_y = np.zeros((batch_size, ) + dis_patch)

                    discriminator.trainable = True

                    # Train the discriminators (original images = real / generated = Fake)
                    d_loss_real = discriminator.train_on_batch(imgs_hr, real_y)
                    d_loss_fake = discriminator.train_on_batch(fake_hr, fake_y)
                    d_loss = 0.5 * np.add(d_loss_real[0], d_loss_fake[0])
                    informations['d_real_loss'].append(d_loss_real[0])
                    informations['d_fake_loss'].append(d_loss_fake[0])
                    informations['d_loss'].append(d_loss)

                    # ------------------
                    #  Train Generator

                    imgs_hr, imgs_lr = data_manager.load_prepared_data(
                        batch_size)

                    discriminator.trainable = False

                    # Extract ground truth image features using pre-trained VGG19 model
                    vgg_y = vgg.predict(imgs_hr)

                    # Train the generators
                    g_loss = adversarial.train_on_batch(
                        imgs_lr, [imgs_hr, vgg_y])
                    informations['g_loss1'].append(g_loss[0])
                    informations['g_loss2'].append(g_loss[1])

                    # If at save interval => save generated image samples
                    if epoch % sample_interval == 0:
                        data_manager.sample_images(generator, epoch, 2)

        return informations

    return train_srgan