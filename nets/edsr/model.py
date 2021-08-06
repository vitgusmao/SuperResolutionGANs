import ipdb
import pandas as pd
import tensorflow as tf
from keras.layers import Add, Lambda, Input
from keras.layers.convolutional import Conv2D
from keras.models import Model
import numpy as np
import copy

from data_manager import ImagesManager, define_image_process, load_datasets
from utils import ProgressBar
from registry import MODEL_REGISTRY

from metrics import psnr, ssim
from lr_schedule import MultiStepLR
from losses import PixelLoss


def pixel_shuffle(scale):
    return lambda x: tf.nn.depth_to_space(x, scale)


def EDSR_Model(
    gt_size, scale, num_filters=64, num_res_blocks=8, res_block_scaling=None
):
    input_shape = int(gt_size / scale)
    x_in = Input(shape=(input_shape, input_shape, 3))
    # x_in = Input(shape=(None, None, 3)) # Isso talvez permita qualquer tamanho?

    x = b = Conv2D(num_filters, 3, padding="same")(x_in)
    for _ in range(num_res_blocks):
        b = res_block(b, num_filters, res_block_scaling)
    b = Conv2D(num_filters, 3, padding="same")(b)
    x = Add()([x, b])

    x = upsample(x, scale, num_filters)
    x = Conv2D(3, 3, padding="same")(x)

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


@MODEL_REGISTRY.register()
def edsr(config):

    image_manager = ImagesManager(config)
    image_manager.initialize_dirs(2, config["epochs"])

    try:
        history_df = pd.read_csv(f"./histories/{config['name']}.csv")
        history = {
            col_name: history_df[col_name].tolist() for col_name in history_df.columns
        }
    except (FileNotFoundError, pd.errors.EmptyDataError):
        history = {"loss": [], "psnr": [], "ssim": []}
    _history = copy.deepcopy(history)

    imgs_config = config["images"]
    train_config = config["train"]

    # define network
    model = EDSR_Model(
        imgs_config["gt_size"],
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
    learning_rate = MultiStepLR(
        train_config["lr"], train_config["lr_steps"], train_config["lr_rate"]
    )
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate,
        beta_1=tf.Variable(train_config["adam_beta1"]),
        beta_2=tf.Variable(train_config["adam_beta2"]),
    )

    # load checkpoint
    checkpoint_dir = "./checkpoints/" + config["name"]
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

            sr = checkpoint.model(lr, training=True)
            loss_value = loss_fn(hr, sr)

        gradients = tape.gradient(loss_value, checkpoint.model.trainable_variables)
        checkpoint.optimizer.apply_gradients(
            zip(gradients, checkpoint.model.trainable_variables)
        )

        return loss_value, sr

    # training loop
    prog_bar = ProgressBar(config["epochs"], checkpoint.step.numpy())
    remain_steps = max(config["epochs"] - checkpoint.step.numpy(), 0)

    for raw_img in train_dataset.take(remain_steps):
        lr, hr = process_image(raw_img)

        checkpoint.step.assign_add(1)
        steps = checkpoint.step.numpy()

        total_loss, sr = train_step(lr, hr)
        prog_bar.update("loss={:.4f}".format(total_loss.numpy()))

        img_psnr = psnr(hr, sr).numpy()
        img_ssim = ssim(hr, sr).numpy()
        if img_psnr.shape[0] == 1:
            img_psnr = img_psnr.squeeze()
            img_ssim = img_ssim.squeeze()
        _history["psnr"].append(img_psnr)
        _history["ssim"].append(img_ssim)
        _history["loss"].append(total_loss.numpy())

        if steps % config["save_steps"] == 0:
            manager.save()
            history = copy.deepcopy(_history)
            print(f"\n>> saved chekpoint file at {manager.latest_checkpoint}.")

        if steps % config["gen_steps"] == 0:
            image_manager.generate_and_save_images(model, steps, 2)

    print(f"\n>> training done for {config['name']}!")

    return history
