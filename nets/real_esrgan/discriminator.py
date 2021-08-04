import tensorflow as tf
from keras.engine.input_layer import Input
from keras.layers.convolutional import Conv2D, UpSampling2D
from tensorflow_addons.layers import SpectralNormalization
from keras.layers import LeakyReLU, Add

keras = tf.keras

def build_u_net_discriminator_sn(input_shape, num_feat=64, skip_connection=True):
    """Defines a U-Net discriminator with spectral normalization (SN)"""

    net_input = Input(input_shape)

    x0 = Conv2D(num_feat, kernel_size=3, stride=1, padding=1)(net_input)
    x0 = LeakyReLU(alpha=0.2)(x0)

    x1 = SpectralNormalization(Conv2D(num_feat * 2, kernel_size=4, strides=2, padding="valid", use_bias=False))(x0)
    x1 = LeakyReLU(alpha=0.2)(x1)
    x2 = SpectralNormalization(Conv2D(num_feat * 4, kernel_size=4, strides=2, padding="valid", use_bias=False))(x1)
    x2 = LeakyReLU(alpha=0.2)(x2)
    x3 = SpectralNormalization(Conv2D(num_feat * 8, kernel_size=4, strides=2, padding="valid", use_bias=False))(x2)
    x3 = LeakyReLU(alpha=0.2)(x3)
    # upsample
    x3 = UpSampling2D(size=2, interpolation='bilinear')(x3)
    x4 = SpectralNormalization(Conv2D(num_feat * 4, kernel_size=3, strides=1, padding="valid", use_bias=False))(x3)
    x4 = LeakyReLU(alpha=0.2)(x4)
    if skip_connection:
        x4 = Add()([x4, x2])

    x4 = UpSampling2D(size=2, interpolation='bilinear')(x4)
    x5 = SpectralNormalization(Conv2D(num_feat * 2, kernel_size=3, strides=1, padding="valid", use_bias=False))(x4)
    x5 = LeakyReLU(alpha=0.2)(x5)
    if skip_connection:
        x5 = Add()([x5, x1])

    x5 = UpSampling2D(size=2, interpolation='bilinear')(x5)
    x6 = SpectralNormalization(Conv2D(num_feat, kernel_size=3, strides=1, padding="valid", use_bias=False))(x5)
    x6 = LeakyReLU(alpha=0.2)(x6)
    if skip_connection:
            x6 = Add()([x6, x0])

    # extra
    output = SpectralNormalization(Conv2D(num_feat, kernel_size=3, strides=1, padding="valid", use_bias=False))(x6)
    output = LeakyReLU(alpha=0.2)(output)
    output = SpectralNormalization(Conv2D(num_feat, kernel_size=3, strides=1, padding="valid", use_bias=False))(output)
    output = LeakyReLU(alpha=0.2)(output)

    output = Conv2D(1, kernel_size=3, strides=1, padding="valid")()

    