from keras import Input
from keras.layers import BatchNormalization, Activation, LeakyReLU, Add, PReLU
from keras.layers.convolutional import Conv2D, UpSampling2D
from keras.models import Model


def build_generator(input_shape, channels=3, generator_filters=64):
    n_residual_blocks = 16

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
    img_lr = Input(shape=input_shape)

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
    gen_hr = Conv2D(channels,
                    kernel_size=9,
                    strides=1,
                    padding='same',
                    activation='tanh')(up_sampling2)

    return Model(img_lr, gen_hr, name='Generator')