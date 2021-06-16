import glob
from os import name

import numpy as np
import tensorflow as tf

from imageio import imread, imwrite
from skimage.transform import resize as imresize

from keras import Input
from keras.callbacks import TensorBoard
from keras.layers import BatchNormalization, Activation, LeakyReLU, Add, Dense, Flatten
from keras.layers.convolutional import Conv2D, UpSampling2D
from keras.models import Model
from keras.optimizers import Adam
from keras_preprocessing.image import img_to_array, load_img

from .metrics.time import timeit_context

# Tratamento do uso de memória da gpu
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

# Common optimizer for all networks
common_optimizer = Adam(0.0002, 0.5)

# Define hyperparameters
data_dir = "datasets/img_align_celeba/*.*"
epochs = 300
BATCH_SIZE = 10

# Shape of low-resolution and high-resolution images
low_resolution_shape = (64, 64, 3)
high_resolution_shape = (256, 256, 3)


def residual_block(x):
    """
    Residual block
    """
    filters = [64, 64]
    kernel_size = 3
    strides = 1
    padding = "same"
    momentum = 0.8
    activation = "relu"

    res = Conv2D(filters=filters[0],
                 kernel_size=kernel_size,
                 strides=strides,
                 padding=padding)(x)
    res = Activation(activation=activation)(res)
    res = BatchNormalization(momentum=momentum)(res)

    res = Conv2D(filters=filters[1],
                 kernel_size=kernel_size,
                 strides=strides,
                 padding=padding)(res)
    res = BatchNormalization(momentum=momentum)(res)

    # Add res and x
    res = Add()([res, x])
    return res


def build_generator():
    """
    Create a generator network using the hyperparameter values defined below
    :return:
    """
    residual_blocks = 16
    momentum = 0.8
    input_shape = (64, 64, 3)

    # Input Layer of the generator network
    input_layer = Input(shape=input_shape)

    # Add the pre-residual block
    gen1 = Conv2D(filters=64,
                  kernel_size=9,
                  strides=1,
                  padding='same',
                  activation='relu')(input_layer)

    # Add 16 residual blocks
    res = residual_block(gen1)
    for i in range(residual_blocks - 1):
        res = residual_block(res)

    # Add the post-residual block
    gen2 = Conv2D(filters=64, kernel_size=3, strides=1, padding='same')(res)
    gen2 = BatchNormalization(momentum=momentum)(gen2)

    # Take the sum of the output from the pre-residual block(gen1) and the post-residual block(gen2)
    gen3 = Add()([gen2, gen1])

    # Add an upsampling block
    gen4 = UpSampling2D(size=2)(gen3)
    gen4 = Conv2D(filters=256, kernel_size=3, strides=1, padding='same')(gen4)
    gen4 = Activation('relu')(gen4)

    # Add another upsampling block
    gen5 = UpSampling2D(size=2)(gen4)
    gen5 = Conv2D(filters=256, kernel_size=3, strides=1, padding='same')(gen5)
    gen5 = Activation('relu')(gen5)

    # Output convolution layer
    gen6 = Conv2D(filters=3, kernel_size=9, strides=1, padding='same')(gen5)
    output = Activation('tanh')(gen6)

    # Keras model
    model = Model(inputs=[input_layer], outputs=[output], name='Generator')
    print(model.summary())
    return model


def build_discriminator():
    """
    Create a discriminator network using the hyperparameter values defined below
    :return:
    """
    leakyrelu_alpha = 0.2
    momentum = 0.8
    input_shape = (256, 256, 3)

    input_layer = Input(shape=input_shape)

    # Add the first convolution block
    dis1 = Conv2D(filters=64, kernel_size=3, strides=1,
                  padding='same')(input_layer)
    # (256, 256, 3) -> ()
    dis1 = LeakyReLU(alpha=leakyrelu_alpha)(dis1)

    # Add the 2nd convolution block
    dis2 = Conv2D(filters=64, kernel_size=3, strides=2, padding='same')(dis1)
    dis2 = LeakyReLU(alpha=leakyrelu_alpha)(dis2)
    dis2 = BatchNormalization(momentum=momentum)(dis2)

    # Add the third convolution block
    dis3 = Conv2D(filters=128, kernel_size=3, strides=1, padding='same')(dis2)
    dis3 = LeakyReLU(alpha=leakyrelu_alpha)(dis3)
    dis3 = BatchNormalization(momentum=momentum)(dis3)

    # Add the fourth convolution block
    dis4 = Conv2D(filters=128, kernel_size=3, strides=2, padding='same')(dis3)
    dis4 = LeakyReLU(alpha=leakyrelu_alpha)(dis4)
    dis4 = BatchNormalization(momentum=0.8)(dis4)

    # Add the fifth convolution block
    dis5 = Conv2D(256, kernel_size=3, strides=1, padding='same')(dis4)
    dis5 = LeakyReLU(alpha=leakyrelu_alpha)(dis5)
    dis5 = BatchNormalization(momentum=momentum)(dis5)

    # Add the sixth convolution block
    dis6 = Conv2D(filters=256, kernel_size=3, strides=2, padding='same')(dis5)
    dis6 = LeakyReLU(alpha=leakyrelu_alpha)(dis6)
    dis6 = BatchNormalization(momentum=momentum)(dis6)

    # Add the seventh convolution block
    dis7 = Conv2D(filters=512, kernel_size=3, strides=1, padding='same')(dis6)
    dis7 = LeakyReLU(alpha=leakyrelu_alpha)(dis7)
    dis7 = BatchNormalization(momentum=momentum)(dis7)

    # Add the eight convolution block
    dis8 = Conv2D(filters=512, kernel_size=3, strides=2, padding='same')(dis7)
    dis8 = LeakyReLU(alpha=leakyrelu_alpha)(dis8)
    dis8 = BatchNormalization(momentum=momentum)(dis8)

    # Add a dense layer
    dis9 = Dense(units=1024)(dis8)
    dis9 = LeakyReLU(alpha=0.2)(dis9)

    # Last dense layer - for classification
    dis9 = Flatten()(dis9)
    output = Dense(units=1, activation='sigmoid')(dis9)
    # output = Reshape((2,1))(output)

    model = Model(inputs=[input_layer], outputs=[output], name='Discriminator')
    print(model.summary())
    return model


def build_vgg():
    model = tf.keras.applications.VGG19(
        include_top=False,
        weights="imagenet",
        input_shape=high_resolution_shape,
        pooling="avg",
    )

    # model = Model(inputs=base_model.input, outputs=[output], name='VGG19')

    model.trainable = False
    print(model.summary())
    return model


def build_adversarial_model(generator, discriminator, input_low, vgg):

    fake_hr_images = generator(input_low)
    fake_features = vgg(fake_hr_images)

    discriminator.trainable = False

    output = discriminator(fake_hr_images)

    model = Model(inputs=[input_low],
                  outputs=[output, fake_features],
                  name='Adversarial')

    # for layer in model.layers:
    #     print(layer.name, layer.trainable)

    print(model.summary())
    return model


vgg = build_vgg()
vgg.compile(loss='mse', optimizer=common_optimizer, metrics=['accuracy'])

discriminator = build_discriminator()
discriminator.compile(loss='mse',
                      optimizer=common_optimizer,
                      metrics=['accuracy'])

generator = build_generator()

# input_high_resolution = Input(shape=high_resolution_shape)
input_low_resolution = Input(shape=low_resolution_shape)

adversarial_model = build_adversarial_model(generator, discriminator,
                                            input_low_resolution, vgg)
adversarial_model.compile(loss=['binary_crossentropy', 'mse'],
                          loss_weights=[1e-3, 1],
                          optimizer=common_optimizer)

# For use of TensorBoard
# tensorboard = TensorBoard(log_dir="logs/".format(time.time()))
# tensorboard.set_model(generator)
# tensorboard.set_model(discriminator)


def sample_images(data_dir, batch_size, high_resolution_shape,
                  low_resolution_shape):
    # Make a list of all images inside the data directory
    all_images = glob.glob(data_dir)

    # Choose a random batch of images
    images_batch = np.random.choice(all_images, size=batch_size)

    low_resolution_images = []
    high_resolution_images = []

    for img in images_batch:
        # Get an ndarray of the current image
        img1 = imread(img, pilmode='RGB')
        img1 = img1.astype(np.float32)

        # Resize the image
        img1_high_resolution = imresize(img1, high_resolution_shape)
        img1_low_resolution = imresize(img1, low_resolution_shape)

        # Do a random flip
        if np.random.random() < 0.5:
            img1_high_resolution = np.fliplr(img1_high_resolution)
            img1_low_resolution = np.fliplr(img1_low_resolution)

        high_resolution_images.append(img1_high_resolution)
        low_resolution_images.append(img1_low_resolution)

    return np.array(high_resolution_images), np.array(low_resolution_images)

with timeit_context('Treino'):
    with tf.device('/gpu:0') as GPU:
        generated_samples = []
        for epoch in range(epochs):
            print("Epoch:{}".format(epoch))

            high_resolution_images, low_resolution_images = sample_images(
                data_dir=data_dir,
                batch_size=BATCH_SIZE,
                low_resolution_shape=low_resolution_shape,
                high_resolution_shape=high_resolution_shape)

            high_resolution_images = high_resolution_images / 127.5 - 1.
            low_resolution_images = low_resolution_images / 127.5 - 1.

            # Generate the images from the noise
            generated_images = generator.predict(low_resolution_images)
            generated_samples.append(generated_images)

            X = np.concatenate((high_resolution_images, generated_images))
            # Create labels
            y = np.zeros(2 * BATCH_SIZE)
            y[:BATCH_SIZE] = 0.9  # One-sided label smoothing

            # Train discriminator on generated images
            discriminator.trainable = True
            d_loss = discriminator.train_on_batch(X, y)

            # Train generator
            y2 = np.ones(BATCH_SIZE)
            discriminator.trainable = False
            g_loss = adversarial_model.train_on_batch(low_resolution_images, y2)

    high_resolution_image, low_resolution_image = sample_images(
        data_dir=data_dir,
        batch_size=1,
        low_resolution_shape=low_resolution_shape,
        high_resolution_shape=high_resolution_shape)

    generated_image = generator.predict(low_resolution_image)

    generated_image = np.fliplr(generated_image)[0]
    high_resolution_image = np.fliplr(high_resolution_image[0])
    imwrite('imgs/{}.jpg'.format('imagem_gerada'),
            generated_image.astype(np.uint8))
    imwrite('imgs/{}.jpg'.format('imagem_original'),
            high_resolution_image.astype(np.uint8))
