import tensorflow as tf
from keras.engine.input_layer import Input
from keras.layers import ReLU, Add
from keras.layers.convolutional import Conv2D
from keras.models import Model
import numpy as np


from utils import ProgressBar
from data_manager import ImagesManager, define_image_process_interpolated, load_datasets
from registry import MODEL_REGISTRY

from losses import PixelLoss


def VDSR_Model(gt_size, channels=3, filters=64):
    input_shape = (gt_size, gt_size, channels)
    net_input = Input(input_shape)

    first_conv = Conv2D(
        filters=filters,
        kernel_size=4,
        kernel_initializer=tf.keras.initializers.random_normal(stddev=np.sqrt(2.0 / 9)),
        use_bias=True,
        bias_initializer=tf.keras.initializers.constant(np.zeros((filters,))),
        strides=1,
        padding="same",
    )(net_input)
    x = ReLU()(first_conv)

    for i in range(18):
        x = Conv2D(
            filters=filters,
            kernel_size=4,
            strides=1,
            kernel_initializer=tf.keras.initializers.random_normal(
                stddev=np.sqrt(2.0 / 9 / filters)
            ),
            use_bias=True,
            bias_initializer=tf.keras.initializers.constant(np.zeros((filters,))),
            padding="same",
        )(x)
        x = ReLU()(x)

    last_conv = Conv2D(
        filters=channels,
        kernel_size=4,
        kernel_initializer=tf.keras.initializers.random_normal(
            stddev=np.sqrt(2.0 / 9 / filters)
        ),
        use_bias=True,
        bias_initializer=tf.keras.initializers.constant(np.zeros((3,))),
        strides=1,
        activation="tanh",
        padding="same",
    )(x)

    output = Add()([net_input, last_conv])

    model = Model(inputs=net_input, outputs=output, name="VDSR")
    print(model.summary())

    return model


@MODEL_REGISTRY.register()
def vdsr(config):

    image_manager = ImagesManager(config)
    image_manager.initialize_dirs(2, config["epochs"])

    imgs_config = config["images"]

    train_config = config["train"]

    # define network
    model = VDSR_Model(
        imgs_config["gt_size"],
        imgs_config["channels"],
        train_config["num_filters"],
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
