from keras.models import Model
from tensorflow.keras.applications import VGG19


def build_vgg(net_input, full_net=False, layers=None):
    # """
    # Builds a pre-trained VGG19 model that outputs image features extracted
    # """

    vgg19 = VGG19(
        include_top=False, weights="imagenet", input_shape=net_input, pooling="avg"
    )

    if not full_net:
        outputs = []
        for tl_layer in layers:
            x = vgg19.get_layer(tl_layer).output
            outputs.append(x)

        model = Model(vgg19.input, outputs, name="VGG19CustomTL")

    else:
        model = vgg19

    model.summary()
    # Make trainable as False
    model.trainable = False

    return model
