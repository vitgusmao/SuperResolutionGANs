from keras.layers import LeakyReLU, BatchNormalization, Dense, Flatten, Input
from keras.layers.convolutional import Conv2D
from keras.models import Model


def Discriminator(gt_size, channels=3, filters=64):
    input_shape = (gt_size, gt_size, channels)
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

    x = Flatten()(x)
    x = Dense(1024, name="dense_0")(x)
    x = LeakyReLU(alpha=0.2, name="leaky_re_lu_final")(x)
    output = Dense(1, name="dense_1", activation="sigmoid")(x)

    model = Model(inputs=net_input, outputs=output, name="Discriminator")
    model.summary(line_length=80)

    return model
