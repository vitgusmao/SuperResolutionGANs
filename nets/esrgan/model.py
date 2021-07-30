import ipdb
import tensorflow as tf

keras = tf.keras

from losses import build_perceptual_vgg
from nets.esrgan.discriminator import build_discriminator
from metrics import image_metrics
from nets.esrgan.rrdbnet import build_rrdbnet
from data_manager import ImagesManager, ImageSequence


class SamplesCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):

        if epoch % 20 == 0:
            self.model.img_m.sample_per_epoch(self.model.gen, epoch, 2)


class ESRGAN(keras.Model):
    def __init__(self, generator, discriminator, image_manager):
        super(ESRGAN, self).__init__()
        self.gen = generator
        self.disc = discriminator
        self.img_m = image_manager
        self.img_m.initialize_dirs(2, 1000)

    def compile(self, gen_optimizer, disc_optimizer, metrics):
        super(ESRGAN, self).compile(run_eagerly=True, metrics=metrics)
        self.gen_optimizer = gen_optimizer
        self.disc_optimizer = disc_optimizer
        self.adv_loss = keras.losses.MeanSquaredError()
        self.content_loss = keras.losses.MeanAbsoluteError()
        self.perceptual_loss = build_perceptual_vgg((256, 256, 3), "block1_conv2")

    def call(self, inputs, training=False, mask=None):
        return self.gen(inputs, training=training, mask=mask)

    def train_step(self, batch_data):

        # x is low res and y is high res
        x_batch, y_batch = batch_data

        # For Enhanced Super Resolution GAN, we need to calculate
        # losses for the generator and discriminator.
        # We will perform the following steps here:
        #
        # 1. Pass low res. images through the generator and get the generated high res. images
        # 2. Pass the generated images in 1) to the discriminator.
        # 3. Calculate the generators total loss (adverserial)
        # 4. Calculate the discriminator loss
        # 5. Update the weights of the generator
        # 6. Update the weights of the discriminator
        # 7. Return the losses in a dictionary

        with tf.GradientTape(persistent=True) as tape:

            self.disc.trainable = True

            # From low res. image generate high res. version
            gen_hr = self.gen(x_batch, training=False)

            # Train the discriminators (original images = real / generated = Fake)
            d_real = self.disc(y_batch, training=True)
            d_fake = self.disc(gen_hr, training=True)

            # Discriminator loss
            real_loss = self.adv_loss(tf.ones_like(d_real), d_real)
            fake_loss = self.adv_loss(tf.zeros_like(d_fake), d_fake)
            disc_loss = (real_loss + fake_loss) * 0.5

            #  Train Generator
            gen_hr = self.gen(x_batch, training=True)
            d_fake = self.disc(gen_hr, training=False)

            # Generator adverserial loss
            adversarial_loss = self.adv_loss(tf.ones_like(d_fake), d_fake)
            content_loss = self.adv_loss(y_batch, gen_hr) * 1e-5
            perceptual_loss = self.perceptual_loss(y_batch, gen_hr)

            # Total generator loss
            total_loss = adversarial_loss + perceptual_loss# + content_loss

        # Get the gradients for the generator
        gen_grads = tape.gradient(total_loss, self.gen.trainable_variables)

        # Get the gradients for the discriminator
        disc_grads = tape.gradient(disc_loss, self.disc.trainable_variables)

        # Update the weights of the generator
        self.gen_optimizer.apply_gradients(
            zip(gen_grads, self.gen.trainable_variables))

        # Update the weights of the discriminator
        self.disc_optimizer.apply_gradients(
            zip(disc_grads, self.disc.trainable_variables))

        return {
            "gen_loss": total_loss,
            "disc_loss": disc_loss,
        }


def compile_and_train(img_shapes, dataset_info, train_args):
    hr_img_shape = img_shapes.get('hr_img_shape')
    lr_img_shape = img_shapes.get('lr_img_shape')

    hr_shape = img_shapes.get('hr_shape')
    lr_shape = img_shapes.get('lr_shape')

    dataset_dir = dataset_info.get('dataset_dir')
    dataset_name = dataset_info.get('dataset_name')

    batch_size = train_args.get('batch_size')
    epochs = train_args.get('epochs')

    net_name = 'ESRGAN'

    image_manager = ImagesManager(dataset_dir, dataset_name, net_name,
                                  hr_img_shape, lr_img_shape)

    image_sequence = ImageSequence(image_manager, 1)

    generator = build_rrdbnet(lr_shape)
    discriminator = build_discriminator(hr_shape)

    # Create enhanced super resolution gan model
    model = ESRGAN(generator=generator,
                          discriminator=discriminator,
                          image_manager=image_manager)

    # Compile the model
    model.compile(
        gen_optimizer=keras.optimizers.Adam(learning_rate=2e-4),
        disc_optimizer=keras.optimizers.Adam(learning_rate=2e-4),
        metrics=image_metrics)

    model.fit(image_sequence,
                     batch_size=batch_size,
                     epochs=epochs,
                    #  use_multiprocessing=True,
                    #  workers=2,
                     callbacks=[SamplesCallback()])
