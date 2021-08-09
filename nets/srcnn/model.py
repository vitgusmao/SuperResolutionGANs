import ipdb
import tensorflow as tf
from keras.layers import Input, Activation
from keras.layers.convolutional import Conv2D
from keras.models import Model
import numpy as np
import pandas as pd
import copy
import os

from utils import ProgressBar, load_history, save_history
from data_manager import ImagesManager, load_datasets, define_image_process_interpolated
from registry import MODEL_REGISTRY

from losses import PixelLoss
from lr_schedule import MultiStepLR
from metrics import psnr, ssim


def SRCNN_Model(gt_size, channels=3, filters=64):
    inputs = Input((gt_size, gt_size, channels), name="img")

    x = Conv2D(filters=filters, kernel_size=9, padding="same")(inputs)
    x = Activation(activation=tf.nn.relu)(x)

    x = Conv2D(filters=int(filters / 2), kernel_size=1, padding="same")(x)
    x = Activation(activation=tf.nn.relu)(x)

    outputs = Conv2D(filters=channels, kernel_size=5, padding="same")(x)

    model = Model(inputs=inputs, outputs=outputs, name="SRCNN_model")
    model.summary()

    return model


@MODEL_REGISTRY.register()
def srcnn(config):

    image_manager = ImagesManager(config)
    image_manager.initialize_dirs(2, config["epochs"])

    imgs_config = config["images"]
    train_config = config["train"]

    # define network
    model = SRCNN_Model(
        imgs_config["gt_size"], imgs_config["channels"], train_config["num_filters"]
    )

    # load dataset
    train_dataset = load_datasets(
        config["datasets"], "train_datasets", config["batch_size"], shuffle=False
    )
    # image_loader = image_manager.get_dataset()
    process_image = define_image_process_interpolated(
        imgs_config["gt_size"], imgs_config["scale"]
    )

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

    history = load_history(config, manager.latest_checkpoint)
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
        history["psnr"].append(img_psnr)
        history["ssim"].append(img_ssim)
        history["loss"].append(total_loss.numpy())

        if steps % config["save_steps"] == 0:
            manager.save()
            save_history(history, config)
            print(f"\n>> saved chekpoint file at {manager.latest_checkpoint}.")

        if steps % config["gen_steps"] == 0:
            image_manager.generate_and_save_images_cnn(model, steps, 2)

    print(f"\n>> training done for {config['name']}!")
