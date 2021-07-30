import ipdb
import tensorflow as tf
from keras.engine.input_layer import Input
from keras.layers import ReLU, Add
from keras.layers.convolutional import Conv2D, UpSampling2D
from keras.models import Model
import numpy as np

keras = tf.keras

from data_manager import ImagesManager, CNNImageSequence


class SamplesCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):

        if epoch % 20 == 0:
            self.model.img_m.sample_per_epoch_cnn(self.model.net, epoch, 2)


class SRCNN(keras.Model):
    def __init__(self, lr_shape=None, hr_shape=None, image_manager=None):
        super(SRCNN, self).__init__()
        self.img_m = image_manager
        self.img_m.initialize_dirs(2, 1000)

        self.channels = 3

        self.net = self.build_model(hr_shape)

    def build_model(self, input_shape):

        net_input = Input(input_shape)

        conv1 = Conv2D(
            filters=64,
            kernel_size=9,
            kernel_initializer=keras.initializers.random_normal(9,
                                                                stddev=1e-3),
            use_bias=True,
            strides=1,
            bias_initializer=keras.initializers.constant(np.zeros((64, ))),
            padding='same',
            activation='relu')(net_input)

        conv2 = Conv2D(
            filters=32,
            kernel_size=1,
            kernel_initializer=keras.initializers.random_normal(1,
                                                                stddev=1e-3),
            use_bias=True,
            strides=1,
            bias_initializer=keras.initializers.constant(np.zeros((32, ))),
            padding='same',
            activation='relu')(conv1)

        conv3 = Conv2D(
            filters=self.channels,
            kernel_size=5,
            kernel_initializer=keras.initializers.random_normal(5,
                                                                stddev=1e-3),
            use_bias=True,
            strides=1,
            bias_initializer=keras.initializers.constant(np.zeros((3, ))),
            padding='same',
        )(conv2)

        output = Add()([net_input, conv3])

        model = Model(inputs=net_input, outputs=output, name='SRCNN')

        print(model.summary())
        return model

    def compile(
        self,
        optimizer,
    ):
        super(SRCNN, self).compile(run_eagerly=True)
        self.optimizer = optimizer

        self.loss_fn = keras.losses.MeanSquaredError()

    def call(self, inputs, training=False, mask=None):
        return self.net(inputs, training=training, mask=mask)

    def train_step(self, batch_data):

        x_batch, y_batch = batch_data

        with tf.GradientTape(persistent=True) as tape:

            gen_img = self.net(x_batch, training=True)

            loss = self.loss_fn(gen_img, y_batch)

        # Get the gradients for the generator
        grads = tape.gradient(loss, self.net.trainable_variables)

        # Update the weights of the generator
        self.optimizer.apply_gradients(zip(grads,
                                           self.net.trainable_variables))

        return {
            "loss": loss,
        }


def train_and_compile():
    # Input shapes
    channels = 3

    lr_height = 64
    lr_width = 64
    lr_img_shape = (lr_height, lr_width)
    lr_shape = (lr_height, lr_width, channels)

    hr_height = lr_height * 4
    hr_width = lr_width * 4
    hr_img_shape = (hr_height, hr_width)
    hr_shape = (hr_height, hr_width, channels)

    dataset_name = 'DIV2K_train_HR'
    dataset_dir = '../datasets/{}/'
    image_manager = ImagesManager(dataset_dir, dataset_name, hr_img_shape,
                                  lr_img_shape)

    image_sequence = CNNImageSequence(dataset_dir, dataset_name, hr_img_shape,
                                      hr_img_shape, 1)

    model = SRCNN(hr_shape=hr_shape, image_manager=image_manager)

    # Compile the model
    model.compile(optimizer=keras.optimizers.SGD(learning_rate=1e-4))

    model.fit(image_sequence,
              batch_size=20,
              epochs=120,
              use_multiprocessing=True,
              workers=2,
              callbacks=[SamplesCallback()])