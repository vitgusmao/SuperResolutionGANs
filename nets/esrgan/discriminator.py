# Based on https://github.com/peteryuX/esrgan-tf2 implementation of https://github.com/xinntao/BasicSR implementation
from keras.layers.convolutional import Conv2D
from keras.models import Model
from keras.layers import (
    LeakyReLU,
    # SyncBatchNormalization,
    # BatchNormalization,
    Dense,
    Flatten,
    Input,
)
import functools
import tensorflow as tf


def _regularizer(weights_decay=5e-4):
    return tf.keras.regularizers.l2(weights_decay)


def _kernel_init(scale=1.0, seed=None):
    """He normal initializer with scale."""
    scale = 2.0 * scale
    return tf.keras.initializers.VarianceScaling(
        scale=scale, mode="fan_in", distribution="truncated_normal", seed=seed
    )


class BatchNormalization(tf.keras.layers.BatchNormalization):
    """
    ref: https://github.com/zzh8829/yolov3-tf2
    """

    def __init__(
        self,
        axis=-1,
        momentum=0.9,
        epsilon=1e-5,
        center=True,
        scale=True,
        name=None,
        **kwargs
    ):
        super(BatchNormalization, self).__init__(
            axis=axis,
            momentum=momentum,
            epsilon=epsilon,
            center=center,
            scale=scale,
            name=name,
            **kwargs
        )

    def call(self, x, training=False):
        if training is None:
            training = tf.constant(False)
        training = tf.logical_and(training, self.trainable)
        return super().call(x, training)


def DiscriminatorVGG128(size, channels, nf=64, wd=0.0, name="Discriminator_VGG_128"):
    
    lrelu_f = functools.partial(LeakyReLU, alpha=0.2)
    conv_k3s1_f = functools.partial(
        Conv2D,
        kernel_size=3,
        strides=1,
        padding="same",
        kernel_initializer=_kernel_init(),
        kernel_regularizer=_regularizer(wd),
    )
    conv_k4s2_f = functools.partial(
        Conv2D,
        kernel_size=4,
        strides=2,
        padding="same",
        kernel_initializer=_kernel_init(),
        kernel_regularizer=_regularizer(wd),
    )
    dese_f = functools.partial(Dense, kernel_regularizer=_regularizer(wd))

    x = inputs = Input(shape=(size, size, channels))

    x = conv_k3s1_f(filters=nf, name="conv0_0")(x)
    x = conv_k4s2_f(filters=nf, use_bias=False, name="conv0_1")(x)
    x = lrelu_f()(BatchNormalization(name="bn0_1")(x))

    x = conv_k3s1_f(filters=nf * 2, use_bias=False, name="conv1_0")(x)
    x = lrelu_f()(BatchNormalization(name="bn1_0")(x))
    x = conv_k4s2_f(filters=nf * 2, use_bias=False, name="conv1_1")(x)
    x = lrelu_f()(BatchNormalization(name="bn1_1")(x))

    x = conv_k3s1_f(filters=nf * 4, use_bias=False, name="conv2_0")(x)
    x = lrelu_f()(BatchNormalization(name="bn2_0")(x))
    x = conv_k4s2_f(filters=nf * 4, use_bias=False, name="conv2_1")(x)
    x = lrelu_f()(BatchNormalization(name="bn2_1")(x))

    x = conv_k3s1_f(filters=nf * 8, use_bias=False, name="conv3_0")(x)
    x = lrelu_f()(BatchNormalization(name="bn3_0")(x))
    x = conv_k4s2_f(filters=nf * 8, use_bias=False, name="conv3_1")(x)
    x = lrelu_f()(BatchNormalization(name="bn3_1")(x))

    x = conv_k3s1_f(filters=nf * 8, use_bias=False, name="conv4_0")(x)
    x = lrelu_f()(BatchNormalization(name="bn4_0")(x))
    x = conv_k4s2_f(filters=nf * 8, use_bias=False, name="conv4_1")(x)
    x = lrelu_f()(BatchNormalization(name="bn4_1")(x))

    x = Flatten()(x)
    x = dese_f(units=100, activation=lrelu_f(), name="linear1")(x)
    out = dese_f(units=1, name="linear2")(x)

    model = Model(inputs, out, name=name)
    model.summary(line_length=80)

    return model
