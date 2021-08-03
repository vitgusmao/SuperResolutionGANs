from keras.engine.input_layer import Input
from keras.layers import LeakyReLU, Add, Lambda, Concatenate
from keras.layers.convolutional import Conv2D, UpSampling2D
from keras.models import Model


def build_residual_dense_block(block_input, filters=64, num_grow_ch=32):
    x1 = Conv2D(
        filters=num_grow_ch,
        kernel_size=3,
        strides=1,
        padding="same",
        # name=f"rdb_1_conv_1",
    )(block_input)
    x1 = LeakyReLU(alpha=0.2)(x1)
    res = Concatenate(axis=3)([block_input, x1])

    x2 = Conv2D(
        filters=num_grow_ch,
        kernel_size=3,
        strides=1,
        padding="same",
        # name=f"rdb_2_conv_2",
    )(res)
    x2 = LeakyReLU(alpha=0.2)(x2)
    res = Concatenate(axis=3)([block_input, x1, x2])

    x3 = Conv2D(
        filters=num_grow_ch,
        kernel_size=3,
        strides=1,
        padding="same",
        # name=f"rdb_3_conv_3",
    )(res)
    x3 = LeakyReLU(alpha=0.2)(x3)
    res = Concatenate(axis=3)([block_input, x1, x2, x3])

    x4 = Conv2D(
        filters=num_grow_ch,
        kernel_size=3,
        strides=1,
        padding="same",
        # name=f"rdb_4_conv_4",
    )(res)
    x4 = LeakyReLU(alpha=0.2)(x4)
    res = Concatenate(axis=3)([block_input, x1, x2, x3, x4])

    x5 = Conv2D(
        filters=filters,
        kernel_size=3,
        strides=1,
        padding="same",
        # name=f"rd_block_{block}_conv_5",
    )(res)

    output = x5 * 0.2
    output = Add()([output, block_input])

    return output


def build_residual_in_residual_dense_block(block_input, filters, num_grow_ch=32):
    x = block_input
    for _ in range(3):
        x = build_residual_dense_block(x, filters, num_grow_ch)

    output = x * 0.2
    output = Add()([output, block_input])

    return output


def build_rrdbn(input_shape, filters=64, num_blocks=23, num_grow_ch=32):

    net_input = Input(input_shape)

    conv_first = Conv2D(filters, kernel_size=3, strides=1, padding="same")(net_input)

    x = conv_first
    for _ in range(num_blocks):
        x = build_residual_in_residual_dense_block(
            x, filters=filters, num_grow_ch=num_grow_ch
        )

    conv_body = Conv2D(filters, kernel_size=3, strides=1, padding="same")(x)

    x = Add()([conv_first, conv_body])

    # upsample
    x = UpSampling2D(size=2, interpolation="nearest")(x)
    x = Conv2D(filters, kernel_size=3, strides=1, padding="same")(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = UpSampling2D(size=2, interpolation="nearest")(x)
    x = Conv2D(filters, kernel_size=3, strides=1, padding="same")(x)
    x = LeakyReLU(alpha=0.2)(x)

    conv_hr = Conv2D(filters, kernel_size=3, strides=1, padding="same")(x)
    output = Conv2D(3, kernel_size=3, strides=1, padding="same", activation="tanh")(
        conv_hr
    )

    model = Model(inputs=net_input, outputs=output, name="RRDBNet")
    print(model.summary())

    return model


def build_rrdbnet(input_shape, filters=64):
    """
    Net data format (batch_size, height, width, channels)
    """

    lrelu = LeakyReLU(alpha=0.2)

    net_input = Input(input_shape)

    first_conv = Conv2D(
        filters=filters,
        kernel_size=3,
        strides=1,
        padding="same",
        name="first_conv",
    )(net_input)

    block = 1
    x = Conv2D(
        filters=filters,
        kernel_size=3,
        strides=1,
        padding="same",
        activation=lrelu,
        name=f"rd_block_{block}_conv_1",
    )(first_conv)
    x = Conv2D(
        filters=filters,
        kernel_size=3,
        strides=1,
        padding="same",
        activation=lrelu,
        name=f"rd_block_{block}_conv_2",
    )(x)
    x = Conv2D(
        filters=filters,
        kernel_size=3,
        strides=1,
        padding="same",
        activation=lrelu,
        name=f"rd_block_{block}_conv_3",
    )(x)
    x = Conv2D(
        filters=filters,
        kernel_size=3,
        strides=1,
        padding="same",
        activation=lrelu,
        name=f"rd_block_{block}_conv_4",
    )(x)
    x = Conv2D(
        filters=filters,
        kernel_size=3,
        strides=1,
        padding="same",
        activation=lrelu,
        name=f"rd_block_{block}_conv_5",
    )(x)

    block += 1
    x = Conv2D(
        filters=filters,
        kernel_size=3,
        strides=1,
        padding="same",
        activation=lrelu,
        name=f"rd_block_{block}_conv_1",
    )(x)
    x = Conv2D(
        filters=filters,
        kernel_size=3,
        strides=1,
        padding="same",
        activation=lrelu,
        name=f"rd_block_{block}_conv_2",
    )(x)
    x = Conv2D(
        filters=filters,
        kernel_size=3,
        strides=1,
        padding="same",
        activation=lrelu,
        name=f"rd_block_{block}_conv_3",
    )(x)
    x = Conv2D(
        filters=filters,
        kernel_size=3,
        strides=1,
        padding="same",
        activation=lrelu,
        name=f"rd_block_{block}_conv_4",
    )(x)
    x = Conv2D(
        filters=filters,
        kernel_size=3,
        strides=1,
        padding="same",
        activation=lrelu,
        name=f"rd_block_{block}_conv_5",
    )(x)

    block += 1
    x = Conv2D(
        filters=filters,
        kernel_size=3,
        strides=1,
        padding="same",
        activation=lrelu,
        name=f"rd_block_{block}_conv_1",
    )(x)
    x = Conv2D(
        filters=filters,
        kernel_size=3,
        strides=1,
        padding="same",
        activation=lrelu,
        name=f"rd_block_{block}_conv_2",
    )(x)
    x = Conv2D(
        filters=filters,
        kernel_size=3,
        strides=1,
        padding="same",
        activation=lrelu,
        name=f"rd_block_{block}_conv_3",
    )(x)
    x = Conv2D(
        filters=filters,
        kernel_size=3,
        strides=1,
        padding="same",
        activation=lrelu,
        name=f"rd_block_{block}_conv_4",
    )(x)
    x = Conv2D(
        filters=filters,
        kernel_size=3,
        strides=1,
        padding="same",
        activation=lrelu,
        name=f"rd_block_{block}_conv_5",
    )(x)

    x = Conv2D(
        filters=filters,
        kernel_size=3,
        strides=1,
        padding="same",
        activation=lrelu,
        name="body_conv",
    )(x)
    x = Add()([first_conv, x])

    # upsample
    x = UpSampling2D(size=2, interpolation="nearest", name="up_sampling_1")(x)
    x = Conv2D(
        filters=filters * 2,
        kernel_size=3,
        strides=1,
        padding="same",
        activation=lrelu,
        name="conv_up_sampling_1",
    )(x)
    x = UpSampling2D(size=2, interpolation="nearest", name="up_sampling_2")(x)
    x = Conv2D(
        filters=filters * 4,
        kernel_size=3,
        strides=1,
        padding="same",
        activation=lrelu,
        name="conv_up_sampling_2",
    )(x)

    x = Conv2D(
        filters=filters,
        kernel_size=3,
        strides=1,
        padding="same",
        activation=lrelu,
        name="conv_hr",
    )(x)
    output = Conv2D(
        filters=3,
        kernel_size=3,
        strides=1,
        padding="same",
        activation="tanh",
        name="last_conv",
    )(x)

    model = Model(inputs=net_input, outputs=output, name="RRDBNet")
    print(model.summary())

    return model
