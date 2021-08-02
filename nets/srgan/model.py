import ipdb
from numpy.core.numeric import full
import tensorflow as tf
from tensorflow.python.keras.engine import training

keras = tf.keras

from losses import build_perceptual_vgg
from nets.srgan.discriminator import build_discriminator
from metrics import psnr, ssim, accuracy
from nets.srgan.generator import build_generator
from data_manager import ImagesManager, ImageSequence


class SamplesCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):

        if epoch % 5 == 0:
            self.model.img_m.generate_and_save_images(self.model.gen, epoch, 2)


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
            y_pred_disc_pred = self.disc(y_pred_gen, training=False)

            # Generator losses
            gan_loss = (
                self.gan_loss(tf.ones_like(y_pred_disc_pred), y_pred_disc_pred) * 1e-3
            )
            perceptual_loss = self.perceptual_loss(y_batch, y_pred_gen) * 1

            # Total generator loss
            total_loss = gan_loss + perceptual_loss

            step_output.update(
                {m.__name__: m(y_batch, y_pred_gen) for m in self.gen_metrics}
            )

        # Get the gradients for the generator
        gen_grads = gen_tape.gradient(total_loss, self.gen.trainable_variables)

        # Update the weights of the generator
        self.gen_optimizer.apply_gradients(zip(gen_grads, self.gen.trainable_variables))

        # Train Discriminator - Real
        with tf.GradientTape() as disc_real_tape:

            y_pred_gen = self.gen(x_batch, training=False)
            y_pred_disc_pred = self.disc(y_pred_gen, training=False)
            y_pred_disc_real = self.disc(y_batch, training=True)
            y_pred = y_pred_disc_real - y_pred_disc_pred

            # Discriminator loss
            real_loss = self.gan_loss(
                tf.ones_like(y_pred_disc_real, dtype=tf.float32) * 0.9, y_pred
            )

            step_output.update(
                {
                    f"{m.__name__}_real": m(
                        tf.ones_like(y_pred_disc_real), y_pred_disc_real
                    )
                    for m in self.disc_metrics
                }
            )

        # Get the gradients for the discriminator
        disc_grads = disc_real_tape.gradient(real_loss, self.disc.trainable_variables)

        # Update the weights of the discriminator
        self.disc_optimizer.apply_gradients(
            zip(disc_grads, self.disc.trainable_variables)
        )

        # Train Discriminator - Fake
        with tf.GradientTape() as disc_fake_tape:

            y_pred_gen = self.gen(x_batch, training=False)
            y_pred_disc_pred = self.disc(y_pred_gen, training=True)
            y_pred_disc_real = self.disc(y_batch, training=False)
            y_pred = y_pred_disc_pred - y_pred_disc_real

            # Discriminator loss
            fake_loss = self.gan_loss(
                tf.zeros_like(y_pred_disc_pred, dtype=tf.float32), y_pred
            )

            step_output.update(
                {
                    f"{m.__name__}_fake": m(
                        tf.zeros_like(y_pred_disc_pred), y_pred_disc_pred
                    )
                    for m in self.disc_metrics
                }
            )

        # Get the gradients for the discriminator
        disc_grads = disc_fake_tape.gradient(fake_loss, self.disc.trainable_variables)

        # Update the weights of the discriminator
        self.disc_optimizer.apply_gradients(
            zip(disc_grads, self.disc.trainable_variables)
        )

        disc_loss = (real_loss + fake_loss) * 0.5

        step_output.update(
            {
                "gen_loss": total_loss,
                "disc_loss": disc_loss,
            }
        )

        return step_output

    def test_step(self, batch_data):
        # Unpack the data
        x_batch, y_batch = batch_data

        y_pred_gen = self.gen(x_batch, training=False)

        y_pred_disc_pred = self.disc(y_pred_gen, training=False)
        y_pred_disc_real = self.disc(y_batch, training=False)

        # Discriminator loss
        disc_pred_loss = self.adv_loss(
            tf.zeros_like(y_pred_disc_pred), y_pred_disc_pred
        )
        disc_real_loss = self.adv_loss(tf.ones_like(y_pred_disc_real), y_pred_disc_real)
        disc_loss = (disc_real_loss + disc_pred_loss) * 0.5

        # Generator loss
        adversarial_loss = self.adv_loss(
            tf.ones_like(y_pred_disc_pred), y_pred_disc_pred
        )
        content_loss = self.adv_loss(y_batch, y_pred_gen) * 1e-2
        perceptual_loss = self.perceptual_loss(y_batch, y_pred_gen)

        # Total generator loss
        total_loss = adversarial_loss + perceptual_loss  # + content_loss

        self.compiled_metrics.update_state(y_batch, y_pred_gen)

        return {m.name: m.result() for m in self.metrics}


def compile_and_train(img_shapes, dataset_info, train_args):
    hr_img_shape = img_shapes.get("hr_img_shape")
    lr_img_shape = img_shapes.get("lr_img_shape")

    hr_shape = img_shapes.get("hr_shape")
    lr_shape = img_shapes.get("lr_shape")

    dataset_dir = dataset_info.get("dataset_dir")
    dataset_name = dataset_info.get("dataset_name")

    batch_size = train_args.get("batch_size")
    epochs = train_args.get("epochs")

    net_name = "SRGAN"

    image_manager = ImagesManager(
        dataset_dir, dataset_name, net_name, hr_img_shape, lr_img_shape
    )
    image_sequence = ImageSequence(image_manager, 1)
    image_manager.initialize_dirs(2, epochs)

    generator = build_generator(lr_shape)
    discriminator = build_discriminator(hr_shape)

    checkpoint_filepath = f"checkpoints/{net_name}/"
    model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor="psnr",
        mode="max",
        save_best_only=True,
    )

    # Create enhanced super resolution gan model
    model = SRGAN(
        generator=generator, discriminator=discriminator, image_manager=image_manager
    )

    # Compile the model
    model.compile(
        gen_optimizer=keras.optimizers.Adam(learning_rate=2e-5),
        disc_optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        gen_metrics=[psnr, ssim],
        disc_metrics=[accuracy],
    )

    try:
        model.load_weights(checkpoint_filepath)
    except Exception:
        pass

    history = model.fit(
        image_sequence,
        batch_size=batch_size,
        epochs=epochs,
        #  use_multiprocessing=True,
        #  workers=2,
        callbacks=[SamplesCallback(), model_checkpoint_callback],
    )
    # print(model.disc(image_sequence[200][1]))
