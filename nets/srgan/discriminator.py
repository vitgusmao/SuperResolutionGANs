# Number of filters in the first layer of G and D

from keras.engine.input_layer import Input
from keras.layers import LeakyReLU, BatchNormalization, Dense
from keras.layers.convolutional import Conv2D
from keras.layers.core import Flatten
from keras.models import Model


def build_discriminator(input_shape, filters=64):

    net_input = Input(input_shape)

    block = 0
    x = Conv2D(
        filters=filters,
        kernel_size=3,
        kernel_initializer="glorot_normal",
        strides=1,
        padding="same",
        use_bias=True,
        name="block_{}_conv_0".format(block),
    )(net_input)
    x = LeakyReLU(alpha=0.2, name="block_{}_leaky_re_lu_1".format(block))(x)

    x = Conv2D(
        filters=filters,
        kernel_size=3,
        strides=2,
        padding="same",
        use_bias=False,
        name="block_{}_conv_1".format(block),
    )(x)
    x = BatchNormalization(name="block_{}_bn_0".format(block))(x)
    x = LeakyReLU(alpha=0.2, name="block_{}_leaky_re_lu_2".format(block))(x)

    block += 1
    x = Conv2D(
        filters=filters * 2,
        kernel_size=3,
        kernel_initializer="glorot_normal",
        strides=1,
        padding="same",
        use_bias=False,
        name="block_{}_conv_0".format(block),
    )(x)
    x = BatchNormalization(name="block_{}_bn_0".format(block))(x)
    x = LeakyReLU(alpha=0.2, name="block_{}_leaky_re_lu_1".format(block))(x)
    x = Conv2D(
        filters=filters * 2,
        kernel_size=3,
        strides=2,
        padding="same",
        use_bias=False,
        name="block_{}_conv_1".format(block),
    )(x)
    x = BatchNormalization(name="block_{}_bn_1".format(block))(x)
    x = LeakyReLU(alpha=0.2, name="block_{}_leaky_re_lu_2".format(block))(x)

    block += 1
    x = Conv2D(
        filters=filters * 4,
        kernel_size=3,
        kernel_initializer="glorot_normal",
        strides=1,
        padding="same",
        use_bias=False,
        name="block_{}_conv_0".format(block),
    )(x)
    x = BatchNormalization(name="block_{}_bn_0".format(block))(x)
    x = LeakyReLU(alpha=0.2, name="block_{}_leaky_re_lu_1".format(block))(x)
    x = Conv2D(
        filters=filters * 4,
        kernel_size=3,
        strides=2,
        padding="same",
        use_bias=False,
        name="block_{}_conv_1".format(block),
    )(x)
    x = BatchNormalization(name="block_{}_bn_1".format(block))(x)
    x = LeakyReLU(alpha=0.2, name="block_{}_leaky_re_lu_2".format(block))(x)

    block += 1
    x = Conv2D(
        filters=filters * 8,
        kernel_size=3,
        kernel_initializer="glorot_normal",
        strides=1,
        padding="same",
        use_bias=False,
        name="block_{}_conv_0".format(block),
    )(x)
    x = BatchNormalization(name="block_{}_bn_0".format(block))(x)
    x = LeakyReLU(alpha=0.2, name="block_{}_leaky_re_lu_1".format(block))(x)
    x = Conv2D(
        filters=filters * 8,
        kernel_size=3,
        strides=2,
        padding="same",
        use_bias=False,
        name="block_{}_conv_1".format(block),
    )(x)
    x = BatchNormalization(name="block_{}_bn_1".format(block))(x)
    x = LeakyReLU(alpha=0.2, name="block_{}_leaky_re_lu_2".format(block))(x)

    # block += 1
    # x = Conv2D(
    #     filters=filters * 8,
    #     kernel_size=3,
    #     kernel_initializer="glorot_normal",
    #     strides=1,
    #     padding="same",
    #     use_bias=False,
    #     name="block_{}_conv_0".format(block),
    # )(x)
    # x = BatchNormalization(name="block_{}_bn_0".format(block))(x)
    # x = LeakyReLU(alpha=0.2, name="block_{}_leaky_re_lu_1".format(block))(x)
    # x = Conv2D(
    #     filters=filters * 8,
    #     kernel_size=4,
    #     kernel_initializer="glorot_normal",
    #     strides=2,
    #     padding="same",
    #     use_bias=False,
    #     name="block_{}_conv_1".format(block),
    # )(x)
    # x = BatchNormalization(name="block_{}_bn_1".format(block))(x)
    # x = LeakyReLU(alpha=0.2, name="block_{}_leaky_re_lu_2".format(block))(x)
    x = Flatten()(x)
    x = Dense(1024, name="dense_0")(x)
    x = LeakyReLU(alpha=0.2, name="leaky_re_lu_final")(x)
    x = Dense(1, name="dense_1", activation="sigmoid")(x)

    model = Model(inputs=net_input, outputs=x, name="Discriminator")
    model.summary()

    return model
