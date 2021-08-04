from losses import PixelLoss
import ipdb
import tensorflow as tf
from keras.layers import Add, Lambda, Input
from keras.layers.convolutional import Conv2D
from keras.models import Model
import numpy as np

from data_manager import ImagesManager, define_image_process, load_datasets
from utils import normalize, denormalize, ProgressBar
from registry import MODEL_REGISTRY


def pixel_shuffle(scale):
    return lambda x: tf.nn.depth_to_space(x, scale)


def EDSR_Model(scale, num_filters=64, num_res_blocks=8, res_block_scaling=None):
    x_in = Input(shape=(None, None, 3))
    # x = Lambda(lambda x: normalize(x.numpy()))(x_in)

    x = b = Conv2D(num_filters, 3, padding="same")(x_in)
    for i in range(num_res_blocks):
        b = res_block(b, num_filters, res_block_scaling)
    b = Conv2D(num_filters, 3, padding="same")(b)
    x = Add()([x, b])

    x = upsample(x, scale, num_filters)
    x = Conv2D(3, 3, padding="same")(x)

    # x = Lambda(lambda x: denormalize(x.numpy()))(x)
    return Model(x_in, x, name="edsr")


def res_block(x_in, filters, scaling):
    x = Conv2D(filters, 3, padding="same", activation="relu")(x_in)
    x = Conv2D(filters, 3, padding="same")(x)
    if scaling:
        x = Lambda(lambda t: t * scaling)(x)
    x = Add()([x_in, x])
    return x


def upsample(x, scale, num_filters):
    def upsample_1(x, factor, **kwargs):
        x = Conv2D(num_filters * (factor ** 2), 3, padding="same", **kwargs)(x)
        return Lambda(pixel_shuffle(scale=factor))(x)

    if scale == 2:
        x = upsample_1(x, 2, name="conv2d_1_scale_2")
    elif scale == 3:
        x = upsample_1(x, 3, name="conv2d_1_scale_3")
    elif scale == 4:
        x = upsample_1(x, 2, name="conv2d_1_scale_2")
        x = upsample_1(x, 2, name="conv2d_2_scale_2")

    return x


# def edsr(config):

#     loss_mean = PixelLoss(config)

#     ckpt_mgr = self.checkpoint_manager
#     ckpt = self.checkpoint

#     self.now = time.perf_counter()

#     for lr, hr in train_dataset.take(steps - ckpt.step.numpy()):
#         ckpt.step.assign_add(1)
#         step = ckpt.step.numpy()

#         loss = self.train_step(lr, hr)
#         loss_mean(loss)

#         if step % evaluate_every == 0:
#             loss_value = loss_mean.result()
#             loss_mean.reset_states()

#             # Compute PSNR on validation dataset
#             psnr_value = self.evaluate(valid_dataset)

#             duration = time.perf_counter() - self.now
#             print(
#                 f"{step}/{steps}: loss = {loss_value.numpy():.3f}, PSNR = {psnr_value.numpy():3f} ({duration:.2f}s)"
#             )

#             if save_best_only and psnr_value <= ckpt.psnr:
#                 self.now = time.perf_counter()
#                 # skip saving checkpoint, no PSNR improvement
#                 continue

#             ckpt.psnr = psnr_value
#             ckpt_mgr.save()

#             self.now = time.perf_counter()


@MODEL_REGISTRY.register()
def edsr(config):

    image_manager = ImagesManager(config)
    image_manager.initialize_dirs(2, config["epochs"])

    imgs_config = config["images"]

    train_config = config["train"]

    # define network
    model = EDSR_Model(
        imgs_config["scale"],
        train_config["num_filters"],
        train_config["num_blocks"],
    )

    # load dataset
    train_dataset = load_datasets(
        config["datasets"], "train_datasets", config["batch_size"], shuffle=False
    )
    # image_loader = image_manager.get_dataset()
    process_image = define_image_process(imgs_config["gt_size"], imgs_config["scale"])

    # define losses function
    loss_fn = PixelLoss(criterion=train_config["criterion"])

    # define optimizer
    learning_rate = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        boundaries=train_config["boundaries"], values=train_config["lr_values"]
    )
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate,
        beta_1=tf.Variable(train_config["adam_beta1"]),
        beta_2=tf.Variable(train_config["adam_beta2"]),
    )

    # load checkpoint
    checkpoint_dir = "./checkpoints/" + config["net"]
    checkpoint = tf.train.Checkpoint(
        step=tf.Variable(0, name="step"),
        optimizer=optimizer,
        model=model,
    )
    manager = tf.train.CheckpointManager(
        checkpoint=checkpoint, directory=checkpoint_dir, max_to_keep=3
    )

    if manager.latest_checkpoint:
        ckpt_status = checkpoint.restore(manager.latest_checkpoint)
        ckpt_status.expect_partial()
        # ckpt_status.assert_consumed()
        print(
            ">> load ckpt from {} at step {}.".format(
                manager.latest_checkpoint, checkpoint.step.numpy()
            )
        )
    else:
        print(">> training from scratch.")

    # define training step function
    @tf.function
    def train_step(lr, hr):
        with tf.GradientTape() as tape:
            lr = tf.cast(lr, tf.float32)
            hr = tf.cast(hr, tf.float32)

            sr = checkpoint.model(lr, training=True)
            loss_value = loss_fn(hr, sr)

        gradients = tape.gradient(loss_value, checkpoint.model.trainable_variables)
        checkpoint.optimizer.apply_gradients(
            zip(gradients, checkpoint.model.trainable_variables)
        )

        return loss_value

    # training loop
    prog_bar = ProgressBar(config["epochs"], checkpoint.step.numpy())
    remain_steps = max(config["epochs"] - checkpoint.step.numpy(), 0)

    for raw_img in train_dataset.take(remain_steps):
        lr, hr = process_image(raw_img)

        checkpoint.step.assign_add(1)
        steps = checkpoint.step.numpy()

        total_loss = train_step(lr, hr)

        prog_bar.update("loss={:.4f}".format(total_loss.numpy()))

        if steps % config["save_steps"] == 0:
            manager.save()
            print(f"\n>> saved chekpoint file at {manager.latest_checkpoint}.")

        if steps % config["gen_steps"] == 0:
            image_manager.generate_and_save_images_cnn(model, steps, 2)

    print(f"\n>> training done for {config['net']}!")

    return model
