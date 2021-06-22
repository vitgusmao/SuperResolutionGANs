from keras.layers.convolutional import Conv2D
from keras.models import Model
from tensorflow.keras.applications import VGG19


def build_vgg(net_input, full_net=False):
    # """
    # Builds a pre-trained VGG19 model that outputs image features extracted
    # """

    vgg19 = VGG19(include_top=False, weights='imagenet', input_shape=net_input)

    if not full_net:
        x = vgg19.get_layer('block1_conv2').output
        x = Conv2D(3, 3, padding='same', name='conv_custom')(x)
        outputs = x

        model = Model([vgg19.input], outputs, name='VGG19CustomTL')

    else:
        model = vgg19

    model.summary()
    # Make trainable as False
    model.trainable = False

    return model