import ipdb
import tensorflow as tf
import numpy as np

keras = tf.keras

from data_manager import ImagesManager, ImageSequence

from nets.srgan.discriminator import build_discriminator
from nets.srgan.generator import build_generator
from losses import build_perceptual_vgg
from metrics import psnr, ssim, accuracy

from registry import MODEL_REGISTRY


class SRGAN(keras.Model):
    def __init__(self, generator, discriminator, image_manager):
        super(SRGAN, self).__init__()
        self.gen = generator
        self.disc = discriminator
        self.img_m = image_manager

    def compile(self, gen_optimizer, disc_optimizer, gen_metrics, disc_metrics):
        super(SRGAN, self).compile(run_eagerly=True)
        self.gen_optimizer = gen_optimizer
        self.disc_optimizer = disc_optimizer

        self.gan_loss = keras.losses.BinaryCrossentropy()
        vgg_input = (
            self.gen.output.shape[1],
            self.gen.output.shape[2],
            self.gen.output.shape[3],
        )
        self.perceptual_loss = build_perceptual_vgg(vgg_input, layer="block5_conv4")

        self.gen_metrics = gen_metrics
        self.disc_metrics = disc_metrics

    def call(self, inputs, training=False, mask=None):
        return self.gen(inputs, training=training, mask=mask)

    def train_step(self, batch_data):

        # x is low res and y is high res
        x_batch, y_batch = batch_data

        step_output = {}
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

        # Train Generator
        with tf.GradientTape() as gen_tape:
            y_pred_gen = self.gen(x_batch, training=True)

            y_pred_disc_real = self.disc(y_batch, training=False)
            y_pred_disc_fake = self.disc(y_pred_gen, training=False)

            y_real = tf.zeros_like(y_pred_disc_real, dtype=tf.float32)
            real_loss = self.gan_loss(y_real, y_pred_disc_real)
            y_fake = tf.ones_like(y_pred_disc_fake, dtype=tf.float32)
            fake_loss = self.gan_loss(y_fake, y_pred_disc_fake)

            # Generator losses
            gan_loss = ((real_loss + fake_loss) / 2) * 1e-3
            perceptual_loss = self.perceptual_loss(y_batch, y_pred_gen) * 1
            total_loss = gan_loss + perceptual_loss

        # Get the gradients for the generator
        gen_grads = gen_tape.gradient(total_loss, self.gen.trainable_variables)

        # Update the weights of the generator
        self.gen_optimizer.apply_gradients(zip(gen_grads, self.gen.trainable_variables))

        # Train Discriminator - Real
        with tf.GradientTape() as disc_tape:
            y_pred_disc_real = self.disc(y_batch, training=True)
            y_pred_disc_fake = self.disc(y_pred_gen, training=False)

            # loss
            y_real = tf.ones_like(
                y_pred_disc_real, dtype=tf.float32
            ) - np.random.uniform(0, 0.1, y_pred_disc_real.shape)
            y_fake = tf.zeros_like(
                y_pred_disc_fake, dtype=tf.float32
            ) + np.random.uniform(0, 0.1, y_pred_disc_fake.shape)

            real_loss = self.gan_loss(y_real, y_pred_disc_real)
            fake_loss = self.gan_loss(y_fake, y_pred_disc_fake)
            disc_loss = (real_loss + fake_loss) * 0.5

        # Get the gradients for the discriminator
        disc_grads = disc_tape.gradient(disc_loss, self.disc.trainable_variables)

        # Update the weights of the discriminator
        self.disc_optimizer.apply_gradients(
            zip(disc_grads, self.disc.trainable_variables)
        )

        # Train Discriminator - Fake
        with tf.GradientTape() as disc_tape:
            y_pred_disc_real = self.disc(y_batch, training=False)
            y_pred_disc_fake = self.disc(y_pred_gen, training=True)

            # loss
            y_real = tf.ones_like(
                y_pred_disc_real, dtype=tf.float32
            ) - np.random.uniform(0, 0.1)
            y_fake = tf.zeros_like(
                y_pred_disc_fake, dtype=tf.float32
            ) + np.random.uniform(0, 0.1)

            real_loss = self.gan_loss(y_real, y_pred_disc_real)
            fake_loss = self.gan_loss(y_fake, y_pred_disc_fake)
            disc_loss = (real_loss + fake_loss) * 0.5

        # Get the gradients for the discriminator
        disc_grads = disc_tape.gradient(disc_loss, self.disc.trainable_variables)

        # Update the weights of the discriminator
        self.disc_optimizer.apply_gradients(
            zip(disc_grads, self.disc.trainable_variables)
        )

        step_output.update(
            {m.__name__: m(y_batch, y_pred_gen) for m in self.gen_metrics}
        )
        step_output.update(
            {
                f"{m.__name__}_real": m(
                    tf.ones_like(y_pred_disc_real), y_pred_disc_real
                )
                for m in self.disc_metrics
            }
        )
        step_output.update(
            {
                f"{m.__name__}_fake": m(
                    tf.zeros_like(y_pred_disc_fake), y_pred_disc_fake
                )
                for m in self.disc_metrics
            }
        )

        step_output.update(
            {
                "gen_loss": total_loss,
                "disc_loss": disc_loss,
            }
        )

        return step_output

    # def test_step(self, batch_data):
    #     # Unpack the data
    #     x_batch, y_batch = batch_data

    #     y_pred_gen = self.gen(x_batch, training=False)

    #     y_pred_disc_pred = self.disc(y_pred_gen, training=False)
    #     y_pred_disc_real = self.disc(y_batch, training=False)

    #     # Discriminator loss
    #     disc_pred_loss = self.adv_loss(
    #         tf.zeros_like(y_pred_disc_pred), y_pred_disc_pred
    #     )
    #     disc_real_loss = self.adv_loss(tf.ones_like(y_pred_disc_real), y_pred_disc_real)
    #     disc_loss = (disc_real_loss + disc_pred_loss) * 0.5

    #     # Generator loss
    #     adversarial_loss = self.adv_loss(
    #         tf.ones_like(y_pred_disc_pred), y_pred_disc_pred
    #     )
    #     content_loss = self.adv_loss(y_batch, y_pred_gen) * 1e-2
    #     perceptual_loss = self.perceptual_loss(y_batch, y_pred_gen)

    #     # Total generator loss
    #     total_loss = adversarial_loss + perceptual_loss  # + content_loss

    #     self.compiled_metrics.update_state(y_batch, y_pred_gen)

    #     return {m.name: m.result() for m in self.metrics}


@MODEL_REGISTRY.register()
def srgan(opts, image_manager):
    imgs_opts = opts.get("images")

    lr_size = imgs_opts.get("lr_size")
    hr_size = imgs_opts.get("hr_size")
    channels = imgs_opts.get("channels")

    lr_shape = (lr_size, lr_size, channels)
    hr_shape = (hr_size, hr_size, channels)

    train_opts = opts.get("train")

    g_opts = train_opts.get("generator")
    g_lr = g_opts.get("lr")

    d_opts = train_opts.get("discriminator")
    d_lr = d_opts.get("lr")

    generator = build_generator(lr_shape)
    discriminator = build_discriminator(hr_shape)

    # Create enhanced super resolution gan model
    model = SRGAN(
        generator=generator, discriminator=discriminator, image_manager=image_manager
    )

    # Compile the model
    model.compile(
        gen_optimizer=keras.optimizers.Adam(learning_rate=g_lr),
        disc_optimizer=keras.optimizers.Adam(learning_rate=d_lr),
        gen_metrics=[psnr, ssim],
        disc_metrics=[accuracy],
    )

    return model
