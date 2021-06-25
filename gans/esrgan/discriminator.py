from keras.engine.input_layer import Input
from keras.layers import LeakyReLU, BatchNormalization, Dense
from keras.layers.convolutional import Conv2D
from keras.layers.core import Flatten
from keras.models import Model


def build_discriminator():
    lrelu_alpha = 0.2
    lrelu = LeakyReLU(alpha=lrelu_alpha)

    disc_input = Input((256, 256, 3))

    block = 0
    x = Conv2D(filters=64,
               kernel_size=3,
               kernel_initializer='glorot_normal',
               strides=1,
               padding='same',
               use_bias=True,
               activation=lrelu,
               name='block_{}_conv_0'.format(block))(disc_input)
    x = Conv2D(filters=64,
               kernel_size=4,
               strides=2,
               padding='same',
               use_bias=False,
               name='block_{}_conv_1'.format(block))(x)
    x = BatchNormalization(name='block_{}_bn_0'.format(block))(x)
    x = LeakyReLU(alpha=lrelu_alpha,
                  name='block_{}_leaky_re_lu_1'.format(block))(x)

    block += 1
    x = Conv2D(filters=64 * 2,
               kernel_size=3,
               kernel_initializer='glorot_normal',
               strides=1,
               padding='same',
               use_bias=False,
               name='block_{}_conv_0'.format(block))(x)
    x = BatchNormalization(name='block_{}_bn_0'.format(block))(x)
    x = LeakyReLU(alpha=lrelu_alpha,
                  name='block_{}_leaky_re_lu_1'.format(block))(x)
    x = Conv2D(filters=64 * 2,
               kernel_size=4,
               strides=2,
               padding='same',
               use_bias=False,
               name='block_{}_conv_1'.format(block))(x)
    x = BatchNormalization(name='block_{}_bn_1'.format(block))(x)
    x = LeakyReLU(alpha=lrelu_alpha,
                  name='block_{}_leaky_re_lu_2'.format(block))(x)

    block += 1
    x = Conv2D(filters=64 * 4,
               kernel_size=3,
               kernel_initializer='glorot_normal',
               strides=1,
               padding='same',
               use_bias=False,
               name='block_{}_conv_0'.format(block))(x)
    x = BatchNormalization(name='block_{}_bn_0'.format(block))(x)
    x = LeakyReLU(alpha=lrelu_alpha,
                  name='block_{}_leaky_re_lu_1'.format(block))(x)
    x = Conv2D(filters=64 * 4,
               kernel_size=4,
               strides=2,
               padding='same',
               use_bias=False,
               name='block_{}_conv_1'.format(block))(x)
    x = BatchNormalization(name='block_{}_bn_1'.format(block))(x)
    x = LeakyReLU(alpha=lrelu_alpha,
                  name='block_{}_leaky_re_lu_2'.format(block))(x)

    block += 1
    x = Conv2D(filters=64 * 8,
               kernel_size=3,
               kernel_initializer='glorot_normal',
               strides=1,
               padding='same',
               use_bias=False,
               name='block_{}_conv_0'.format(block))(x)
    x = BatchNormalization(name='block_{}_bn_0'.format(block))(x)
    x = LeakyReLU(alpha=lrelu_alpha,
                  name='block_{}_leaky_re_lu_1'.format(block))(x)
    x = Conv2D(filters=64 * 8,
               kernel_size=4,
               strides=2,
               padding='same',
               use_bias=False,
               name='block_{}_conv_1'.format(block))(x)
    x = BatchNormalization(name='block_{}_bn_1'.format(block))(x)
    x = LeakyReLU(alpha=lrelu_alpha,
                  name='block_{}_leaky_re_lu_2'.format(block))(x)

    block += 1
    x = Conv2D(filters=64 * 8,
               kernel_size=3,
               kernel_initializer='glorot_normal',
               strides=1,
               padding='same',
               use_bias=False,
               name='block_{}_conv_0'.format(block))(x)
    x = BatchNormalization(name='block_{}_bn_0'.format(block))(x)
    x = LeakyReLU(alpha=lrelu_alpha,
                  name='block_{}_leaky_re_lu_1'.format(block))(x)
    x = Conv2D(filters=64 * 8,
               kernel_size=4,
               kernel_initializer='glorot_normal',
               strides=2,
               padding='same',
               use_bias=False,
               name='block_{}_conv_1'.format(block))(x)
    x = BatchNormalization(name='block_{}_bn_1'.format(block))(x)
    x = LeakyReLU(alpha=lrelu_alpha,
                  name='block_{}_leaky_re_lu_2'.format(block))(x)

    x = Dense(100, name='dense_0')(x)
    x = LeakyReLU(alpha=lrelu_alpha, name='leaky_re_lu_final')(x)
    x = Dense(1, name='dense_1', activation='sigmoid')(x)

    model = Model(inputs=disc_input, outputs=x, name='Discriminator')
    model.summary()

    return model
