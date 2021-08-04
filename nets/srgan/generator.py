from keras import Input
from keras.layers import BatchNormalization, Add, PReLU
from keras.layers.convolutional import Conv2D, UpSampling2D
from keras.models import Model


def ResidualBlock(layer_input, filters):
    x = Conv2D(filters=filters, kernel_size=3, strides=1, padding="same")(layer_input)
    x = BatchNormalization(momentum=0.5)(x)
    x = PReLU()(x)
    x = Conv2D(filters, kernel_size=3, strides=1, padding="same")(x)
    x = BatchNormalization(momentum=0.5)(x)
    x = Add()([layer_input, x])
    return x


def UpSample2D(layer_input):
    x = Conv2D(256, kernel_size=3, strides=1, padding="same")(layer_input)
    x = UpSampling2D(size=2)(x)
    x = PReLU()(x)
    return x


def RB_Model(gt_size, scale, channels=3, generator_filters=64, num_blocks=16):
    size = int(gt_size / scale)

    net_input = Input(shape=[size, size, channels])

    conv1 = Conv2D(64, kernel_size=9, strides=1, padding="same")(net_input)
    conv1 = PReLU()(conv1)

    residual_blocks = conv1
    for _ in range(num_blocks):
        residual_blocks = ResidualBlock(residual_blocks, generator_filters)

    # Post-residual block
    conv2 = Conv2D(64, kernel_size=3, strides=1, padding="same")(residual_blocks)
    conv2 = BatchNormalization(momentum=0.5)(conv2)
    conv2 = Add()([conv1, conv2])

    # Upsampling
    up1 = UpSample2D(conv2)
    up2 = UpSample2D(up1)

    # Generate high resolution output
    output = Conv2D(
        channels, kernel_size=9, strides=1, padding="same", activation="tanh"
    )(up2)

    model = Model(inputs=net_input, outputs=output, name="Generator")
    model.summary(line_length=80)

    return model
