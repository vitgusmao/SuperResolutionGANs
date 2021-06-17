import glob
import os
import time
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

optimizer = Adam(0.0002, 0.5)

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

    def load_data(self, batch_size, is_testing=False):
        all_images = self.all_train_images

        if is_testing:
            all_images = self.all_test_images
            images_batch = all_images[1000:(1000 + batch_size)]
        else:
            # Choose a random batch of images
            images_batch_indexes = np.random.randint(0, len(all_images),
                                                     batch_size)
            images_batch = []
            for i in images_batch_indexes:
                images_batch.append(all_images[i])
                del all_images[i]

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
    input_data = (input_data * 127.5) + 127.5
    return input_data.astype(np.uint8)


def build_vgg():
    # """
    # Builds a pre-trained VGG19 model that outputs image features extracted at the
    # third block of the model
    # """
    vgg19 = VGG19(include_top=False, weights='imagenet', input_shape=hr_shape)
    vgg19.trainable = False
    # Make trainable as False
    for l in vgg19.layers:
        l.trainable = False
    model = Model(inputs=vgg19.input,
                  outputs=vgg19.get_layer('block3_conv4').output)
    model.trainable = False

    return model


vgg_model = build_vgg()


def vgg_loss(y_true, y_pred):
    return K.mean(K.square(vgg_model(y_true) - vgg_model(y_pred)))


def build_generator():
    def residual_block(layer_input, filters):
        """Residual block described in paper"""
        block = Conv2D(filters=filters,
                       kernel_size=3,
                       strides=1,
                       padding="same")(layer_input)
        block = BatchNormalization(momentum=0.5)(block)
        block = PReLU(alpha_initializer='zeros',
                      alpha_regularizer=None,
                      alpha_constraint=None,
                      shared_axes=[1, 2])(block)
        block = Conv2D(filters, kernel_size=3, strides=1,
                       padding='same')(block)
        block = BatchNormalization(momentum=0.5)(block)
        block = Add()([layer_input, block])
        return block

    def deconv2d(layer_input):
        """Layers used during upsampling"""
        up_sampling = Conv2D(256, kernel_size=3, strides=1,
                             padding='same')(layer_input)
        up_sampling = UpSampling2D(size=2)(up_sampling)
        up_sampling = LeakyReLU(alpha=0.2)(up_sampling)
        return up_sampling

    # Low resolution image input
    img_lr = Input(shape=lr_shape)

    # Pre-residual block
    conv1 = Conv2D(64, kernel_size=9, strides=1, padding='same')(img_lr)
    conv1 = PReLU(alpha_initializer='zeros',
                  alpha_regularizer=None,
                  alpha_constraint=None,
                  shared_axes=[1, 2])(conv1)

    # Propogate through residual blocks
    residual_blocks = residual_block(conv1, generator_filters)
    for _ in range(n_residual_blocks - 1):
        residual_blocks = residual_block(residual_blocks, generator_filters)

    # Post-residual block
    conv2 = Conv2D(64, kernel_size=3, strides=1,
                   padding='same')(residual_blocks)
    conv2 = BatchNormalization(momentum=0.5)(conv2)
    conv2 = Add()([conv1, conv2])

    # Upsampling
    up_sampling1 = deconv2d(conv2)
    up_sampling2 = deconv2d(up_sampling1)

    # Generate high resolution output
    gen_hr = Conv2D(
        channels,
        kernel_size=9,
        strides=1,
        padding='same',
    )(up_sampling2)
    gen_hr = Activation('tanh')(gen_hr)

    return Model(img_lr, gen_hr)


def build_discriminator():
    def dis_block(layer_input, filters, strides=1, bn=True):
        """Discriminator layer"""
        dis = Conv2D(filters, kernel_size=3, strides=strides,
                     padding='same')(layer_input)
        if bn:
            dis = BatchNormalization(momentum=0.5)(dis)
        dis = LeakyReLU(alpha=0.2)(dis)
        return dis

    # Input img
    dis_input = Input(shape=hr_shape)

    dis = dis_block(dis_input, discriminator_filters, bn=False)
    dis = dis_block(dis, discriminator_filters, strides=2)
    dis = dis_block(dis, discriminator_filters * 2)
    dis = dis_block(dis, discriminator_filters * 2, strides=2)
    dis = dis_block(dis, discriminator_filters * 4)
    dis = dis_block(dis, discriminator_filters * 4, strides=2)
    dis = dis_block(dis, discriminator_filters * 8)
    dis = dis_block(dis, discriminator_filters * 8, strides=2)

    dis = Flatten()(dis)
    dis = Dense(discriminator_filters * 16)(dis)
    dis = LeakyReLU(alpha=0.2)(dis)

    validity = Dense(1)(dis)
    validity = Activation('sigmoid')(validity)

    return Model(dis_input, validity)


def sample_images(epoch=None, is_testing=False):
    os.makedirs('imgs/%s' % dataset_name, exist_ok=True)

    hr_imgs, lr_imgs = data_loader.load_data(batch_size=2, is_testing=True)
    hr_fakes = generator.predict(lr_imgs)

    hr_fakes = denormalize(hr_fakes)

    if not is_testing:
        for hr_gen in hr_fakes:
            imwrite(
                'imgs/{}/{}_{}.jpg'.format(dataset_name, epoch, 'generated'),
                hr_gen.astype(np.uint8))

    else:
        for hr_img, lr_img in zip(hr_imgs, lr_imgs):
            imwrite('imgs/{}/{}.jpg'.format(dataset_name, 'hr'),
                    hr_img.astype(np.uint8))
            imwrite('imgs/{}/{}.jpg'.format(dataset_name, 'lr'),
                    lr_img.astype(np.uint8))


# Build the generator
generator = build_generator()
generator.compile(loss=vgg_loss, optimizer=optimizer)

discriminator = build_discriminator()
discriminator.compile(loss='binary_crossentropy',
                      optimizer=optimizer,
                      metrics=['accuracy'])

# low res. images
img_lr = Input(shape=lr_shape)

# Generate high res. version from low res.
fake_hr = generator(img_lr)

# For the adversarial model we will only train the generator
discriminator.trainable = False

# Discriminator determines validity of generated high res. images
validity = discriminator(fake_hr)

adversarial = Model([img_lr], [fake_hr, validity])
adversarial.compile(loss=[vgg_loss, 'binary_crossentropy', 'mse'],
                    loss_weights=[1.0, 1e-3, 1.0],
                    optimizer=optimizer)
# Remover mse em testes futuros

epochs = 12750
batch_size = 1
sample_interval = 50

sample_images(is_testing=True)

with time_context('treino total'):
    with tf.device('/gpu:0') as GPU:
        for epoch in range(epochs):
            with time_context('tempo de execução da época {}'.format(epoch)):

                # ----------------------
                #  Train Discriminator
                # ----------------------

                # Sample images and their conditioning counterparts
                imgs_hr, imgs_lr = data_loader.load_data(batch_size)
                imgs_hr = normalize(imgs_hr)
                imgs_lr = normalize(imgs_lr)

                # From low res. image generate high res. version
                fake_hr = generator.predict(imgs_lr)

                real_y = np.ones(
                    batch_size) - np.random.random_sample(batch_size) * 0.2
                fake_y = np.random.random_sample(batch_size) * 0.2

                discriminator.trainable = True

                # Train the discriminators (original images = real / generated = Fake)
                d_loss_real = discriminator.train_on_batch(imgs_hr, real_y)
                d_loss_fake = discriminator.train_on_batch(fake_hr, fake_y)
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
                real_y = np.ones(
                    batch_size) - np.random.random_sample(batch_size) * 0.2

                # # Extract ground truth image features using pre-trained VGG19 model
                # image_features = vgg.predict(imgs_hr)

                # Train the generators
                g_loss = adversarial.train_on_batch(imgs_lr, [imgs_hr, real_y])

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                sample_images(epoch)