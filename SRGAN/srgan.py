import numpy as np
import tensorflow as tf

from keras import backend as K
from keras import Input
from keras.models import Model

from data_manager import DataManager
from measures.time_measure import time_context
from optimizers import get_adam_optimizer
from SRGAN.discriminator import build_discriminator
from SRGAN.generator import build_generator



# Input shape
channels = 3
lr_height = 64
lr_width = 64
lr_shape = (lr_height, lr_width, channels)
hr_height = lr_height * 4
hr_width = lr_width * 4
hr_shape = (hr_height, hr_width, channels)

# Number of residual blocks in the generator
n_residual_blocks = 16

optimizer = get_adam_optimizer(lr=1e-4, beta_1=5, amsgrad=True, epsilon=1e-08)

dataset_name = 'img_align_celeba'
dataset_dir = '../datasets/{}/'

data_manager = DataManager()


vgg.trainable = False
vgg.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])


def vgg_loss(y_true, y_pred):
    return K.mean(K.square(vgg_model(y_true) - vgg_model(y_pred)))


def build_srgan():
    # Build the generator
    generator = build_generator()
    generator.compile(loss=vgg_loss, optimizer=optimizer)

    discriminator = build_discriminator()
    discriminator.compile(loss=['binary_crossentropy'],
                          optimizer=optimizer,
                          metrics=['accuracy'])

    discriminator.trainable = False

    img_input = Input(shape=lr_shape)

    gen_hr = generator(img_input)

    validity_output = discriminator(gen_hr)

    adversarial = Model(inputs=img_input, outputs=[gen_hr, validity_output])
    adversarial.compile(loss=[vgg_loss, 'binary_crossentropy'],
                        loss_weights=[1.0, 1e-3],
                        optimizer=optimizer)

    epochs = 300
    batch_size = 8
    sample_interval = 10

    data_manager.initialize_dirs(2)

    with time_context('treino total'):
        with tf.device('/gpu:0') as GPU:
            for epoch in range(epochs):
                # ----------------------
                #  Train Discriminator

                imgs_hr, imgs_lr = data_manager.load_prepared_data(batch_size)

                fake_hr = generator.predict(imgs_lr)

                real_y = np.ones(
                    batch_size) - np.random.random_sample(batch_size) * 0.1
                fake_y = np.random.random_sample(batch_size)

                discriminator.trainable = True

                # Train the discriminators (original images = real / generated = Fake)
                d_loss_real = discriminator.train_on_batch(imgs_hr, real_y)
                d_loss_fake = discriminator.train_on_batch(fake_hr, fake_y)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                # ------------------
                #  Train Generator

                imgs_hr, imgs_lr = data_manager.load_prepared_data(batch_size)

                discriminator.trainable = False

                real_y = np.ones(
                    batch_size) - np.random.random_sample(batch_size) * 0.2

                # Extract ground truth image features using pre-trained VGG19 model
                vgg_y = vgg.predict(imgs_hr)

                # Train the generators
                g_loss = adversarial.train_on_batch(imgs_lr, [imgs_hr, vgg_y])

                # If at save interval => save generated image samples
                if epoch % sample_interval == 0:
                    data_manager.sample_images(generator, epoch, 2)