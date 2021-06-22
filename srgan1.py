from ESRGAN.losses import l1_loss
import glob
import os
import time
import ipdb
from keras.layers.core import Flatten

import numpy as np
import tensorflow as tf

from contextlib import contextmanager

from imageio import imread, imwrite
from keras import Input
from keras import backend as K
from keras_preprocessing.image import img_to_array, load_img
from keras.layers import BatchNormalization, Activation, LeakyReLU, Add, Dense, PReLU
from keras.layers.convolutional import Conv2D, UpSampling2D
from tensorflow.keras.applications import VGG19
from keras.models import Model, load_model
from keras.optimizers import Adam
from skimage.transform import resize as imresize


@contextmanager
def time_context(name):
    start_time = time.time()
    yield
    elapsed_time = time.time() - start_time
    print('[{}] terminou em {} ms'.format(name, int(elapsed_time * 1000)))


# Input shape
channels = 3
lr_height = 64  # Low resolution height
lr_width = 64  # Low resolution width
lr_shape = (lr_height, lr_width, channels)
hr_height = lr_height * 4  # High resolution height
hr_width = lr_width * 4  # High resolution width
hr_shape = (hr_height, hr_width, channels)

# Number of residual blocks in the generator
n_residual_blocks = 16

optimizer = Adam(lr=1E-4, epsilon=1e-08)

# Calculate output shape of D (PatchGAN)
patch = int(hr_height / 2**4)
disc_patch = (patch, patch, 1)

# Number of filters in the first layer of G and D
generator_filters = 64
discriminator_filters = 64

dataset_name = 'img_align_celeba'
dataset_dir = 'datasets/{}/'.format(dataset_name)


class DataLoader:
    def __init__(self):
        # Make a list of all images inside the data directory
        self.all_train_images = glob.glob('{}*.*'.format(dataset_dir))
        self.all_test_images = glob.glob('{}*.*'.format(dataset_dir))

        self.test_random_indexes = np.random.randint(0,
                                                     len(self.all_test_images),
                                                     len(self.all_test_images))

    def load_data(self, batch_size, is_testing=False):

        # Choose a random batch of images
        if is_testing:
            all_images = self.all_test_images
            images_batch_indexes = self.test_random_indexes[:batch_size]
        else:
            all_images = self.all_train_images
            images_batch_indexes = np.random.randint(0, len(all_images),
                                                     batch_size)

        images_batch = []
        for i in images_batch_indexes:
            images_batch.append(all_images[i])
            if not is_testing:
                del all_images[i]

        # ipdb.set_trace()

        lr_images = []
        hr_images = []

        for img in images_batch:
            # Get an ndarray of the current image
            img = imread(img, pilmode='RGB')
            img = img.astype(np.float32)

            # Resize the image
            high_resolution_img = imresize(img, hr_shape)
            low_resolution_img = imresize(img, lr_shape)

            # # Do a random flip
            # if np.random.random() < 0.5:
            #     high_resolution_img = np.fliplr(high_resolution_img)
            #     low_resolution_img = np.fliplr(low_resolution_img)

            hr_images.append(high_resolution_img)
            lr_images.append(low_resolution_img)

        return np.array(hr_images), np.array(lr_images)


data_loader = DataLoader()


def preprocess_HR(x):
    return np.divide(x.astype(np.float32), 127.5) - np.ones_like(
        x, dtype=np.float32)


def deprocess_HR(x):
    x = (x + 1) * 127.5
    return x.astype(np.uint8)


def preprocess_LR(x):
    return np.divide(x.astype(np.float32), 255.)


def deprocess_LR(x):
    x = np.clip(x * 255, 0, 255)
    return x


def normalize(input_data):
    return (input_data.astype(np.float32) - 127.5) / 127.5


def denormalize(input_data):
    input_data = (input_data + 1) * 127.5
    return input_data.astype(np.uint8)


def build_vgg():
    # """
    # Builds a pre-trained VGG19 model that outputs image features extracted at the
    # third block of the model
    # """

    base_model = VGG19(
        weights="imagenet",
        include_top=False,
        input_shape=hr_shape,
    )
    base_model.trainable = False

    inputs = Input(shape=hr_shape)

    x = base_model(inputs, training=False)

    outputs = Conv2D(256, (3, 3),
                     activation='relu',
                     padding='same',
                     name='custom_block_conv')(x)

    model = Model(inputs, outputs)

    return model


def build_generator():
    def residual_block(layer_input, filters):
        """Residual block described in paper"""
        block = Conv2D(filters, kernel_size=3, strides=1,
                       padding='same')(layer_input)
        block = Activation('relu')(block)
        block = BatchNormalization(momentum=0.8)(block)
        block = Conv2D(filters, kernel_size=3, strides=1,
                       padding='same')(block)
        block = BatchNormalization(momentum=0.8)(block)
        block = Add()([block, layer_input])
        return block

    def deconv2d(layer_input):
        """Layers used during upsampling"""
        up_sampling = UpSampling2D(size=2)(layer_input)
        up_sampling = Conv2D(256, kernel_size=3, strides=1,
                             padding='same')(up_sampling)
        up_sampling = Activation('relu')(up_sampling)
        return up_sampling

    # Low resolution image input
    img_lr = Input(shape=lr_shape)

    # Pre-residual block
    conv1 = Conv2D(64, kernel_size=9, strides=1, padding='same')(img_lr)
    conv1 = Activation('relu')(conv1)

    # Propogate through residual blocks
    residual_blocks = residual_block(conv1, generator_filters)
    for _ in range(n_residual_blocks - 1):
        residual_blocks = residual_block(residual_blocks, generator_filters)

    # Post-residual block
    conv2 = Conv2D(64, kernel_size=3, strides=1,
                   padding='same')(residual_blocks)
    conv2 = BatchNormalization(momentum=0.8)(conv2)
    conv2 = Add()([conv2, conv1])

    # Upsampling
    up_sampling1 = deconv2d(conv2)
    up_sampling2 = deconv2d(up_sampling1)

    # Generate high resolution output
    gen_hr = Conv2D(channels,
                    kernel_size=9,
                    strides=1,
                    padding='same',
                    activation='tanh')(up_sampling2)

    return Model(img_lr, gen_hr)


def build_discriminator():
    def dis_block(layer_input, filters, strides=1, bn=True):
        """Discriminator layer"""
        dis = Conv2D(filters, kernel_size=3, strides=strides,
                     padding='same')(layer_input)
        dis = LeakyReLU(alpha=0.2)(dis)
        if bn:
            dis = BatchNormalization(momentum=0.8)(dis)
        return dis

    # Input img
    dis0 = Input(shape=hr_shape)

    dis1 = dis_block(dis0, discriminator_filters, bn=False)
    dis2 = dis_block(dis1, discriminator_filters, strides=2)
    dis3 = dis_block(dis2, discriminator_filters * 2)
    dis4 = dis_block(dis3, discriminator_filters * 2, strides=2)
    dis5 = dis_block(dis4, discriminator_filters * 4)
    dis6 = dis_block(dis5, discriminator_filters * 4, strides=2)
    dis7 = dis_block(dis6, discriminator_filters * 8)
    dis8 = dis_block(dis7, discriminator_filters * 8, strides=2)

    dis9 = Dense(discriminator_filters * 16)(dis8)
    dis10 = LeakyReLU(alpha=0.2)(dis9)
    validity = Dense(1, activation='sigmoid')(dis10)

    return Model(dis0, validity)


def sample_images(epoch=None, create_dirs=False):
    testing_batch_size = 2
    os.makedirs('imgs/%s' % dataset_name, exist_ok=True)
    if create_dirs:
        for i in range(testing_batch_size):
            os.makedirs('imgs/{}/{}'.format(dataset_name, i), exist_ok=True)

    hr_imgs, lr_imgs = data_loader.load_data(batch_size=testing_batch_size,
                                             is_testing=True)
    hr_fakes = generator.predict(lr_imgs)

    hr_fakes = denormalize(hr_fakes)

    if not create_dirs:
        for index, hr_gen in zip(range(len(hr_fakes)), hr_fakes):
            imwrite(
                'imgs/{}/{}/{}_{}.jpg'.format(dataset_name, index, epoch,
                                              'generated'),
                hr_gen.astype(np.uint8))

    else:
        for index, (hr_img, lr_img) in zip(range(len(hr_imgs)),
                                           zip(hr_imgs, lr_imgs)):
            imwrite(
                'imgs/{}/{}/{}.jpg'.format(dataset_name, index,
                                           '?0_high_resolution'),
                hr_img.astype(np.uint8))
            imwrite(
                'imgs/{}/{}/{}.jpg'.format(dataset_name, index,
                                           '?0_low_resolution'),
                lr_img.astype(np.uint8))


# We use a pre-trained VGG19 model to extract image features from the high resolution
# and the generated high resolution images and minimize the mse between them
vgg = build_vgg()
vgg.trainable = False
vgg.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])


def vgg_loss(y_true, y_pred):
    return K.mean(K.square(vgg(y_true) - vgg(y_pred)))


# Build the generator
generator = build_generator()
generator.compile(loss=l1_loss, optimizer=optimizer)

# Build and compile the discriminator
discriminator = build_discriminator()
discriminator.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])

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

sample_images(create_dirs=True)

epochs = 300
batch_size = 1
sample_interval = 50

with time_context('treino total'):
    with tf.device('/gpu:0') as GPU:
        for epoch in range(epochs):
            # ----------------------
            #  Train Discriminator
            # ----------------------

            # Sample images and their conditioning counterparts
            imgs_hr, imgs_lr = data_loader.load_data(batch_size)
            imgs_hr = normalize(imgs_hr)
            imgs_lr = normalize(imgs_lr)

            # From low res. image generate high res. version
            fake_hr = generator.predict(imgs_lr)

            valid = np.ones((batch_size, ) + disc_patch)
            fake = np.zeros((batch_size, ) + disc_patch)

            discriminator.trainable = True

            # Train the discriminators (original images = real / generated = Fake)
            d_loss_real = discriminator.train_on_batch(imgs_hr, valid)
            d_loss_fake = discriminator.train_on_batch(fake_hr, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ------------------
            #  Train Generator
            # ------------------

            # Sample images and their conditioning counterparts
            imgs_hr, imgs_lr = data_loader.load_data(batch_size)
            imgs_hr = normalize(imgs_hr)
            imgs_lr = normalize(imgs_lr)

            discriminator.trainable = False

            # The generators want the discriminators to label the generated images as real
            valid = np.ones((batch_size, ) + disc_patch)

            # Extract ground truth image features using pre-trained VGG19 model
            image_features = vgg.predict(imgs_hr)

            # Train the generators
            g_loss = adversarial.train_on_batch([imgs_lr, imgs_hr],
                                                [valid, image_features])

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                sample_images(epoch)