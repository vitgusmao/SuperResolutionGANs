import tensorflow as tf

from metrics import psnr, ssim
from nets.esrgan.rrdbnet import RRDB_Model
from lr_schedule import MultiStepLR
from losses import PixelLoss
from utils import ProgressBar
from registry import MODEL_REGISTRY
from data_manager import define_image_process, load_datasets, ImagesManager


@MODEL_REGISTRY.register()
def psnr_pretrain(config):
    
    image_manager = ImagesManager(config)
    image_manager.initialize_dirs(2, config["epochs"])

    imgs_config = config["images"]

    train_config = config["train"]

    # define network
    model = RRDB_Model(
        imgs_config["gt_size"],
        imgs_config["scale"],
        imgs_config["channels"],
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
    pixel_loss_fn = PixelLoss(criterion=train_config["pixel_criterion"])

    # define optimizer
    learning_rate = MultiStepLR(
        train_config["lr"], train_config["lr_steps"], train_config["lr_rate"]
    )
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate,
        beta_1=tf.Variable(train_config["adam_beta1"]),
        beta_2=tf.Variable(train_config["adam_beta2"]),
    )
    optimizer.iterations

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
            sr = model(lr, training=True)

            losses = {}
            losses["reg"] = tf.reduce_sum(model.losses)
            losses["pixel"] = train_config["pixel_weight"] * pixel_loss_fn(hr, sr)
            total_loss = tf.add_n([l for l in losses.values()])

        grads = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        # {m.__name__: m(hr, sr) for m in c_metrics}

        return total_loss, losses

    # training loop
    summary_writer = tf.summary.create_file_writer("./logs/" + config["net"])
    prog_bar = ProgressBar(config["epochs"], checkpoint.step.numpy())
    remain_steps = max(config["epochs"] - checkpoint.step.numpy(), 0)

    for raw_img in train_dataset.take(remain_steps):
        lr, hr = process_image(raw_img)

        checkpoint.step.assign_add(1)
        steps = checkpoint.step.numpy()

        total_loss, losses = train_step(lr, hr)

        prog_bar.update(
            "loss={:.4f}, lr={:.1e}".format(
                total_loss.numpy(), optimizer.lr(steps).numpy()
            )
        )

        if steps % 10 == 0:
            with summary_writer.as_default():
                tf.summary.scalar("loss/total_loss", total_loss, step=steps)
                for k, l in losses.items():
                    tf.summary.scalar("loss/{}".format(k), l, step=steps)
                tf.summary.scalar("learning_rate", optimizer.lr(steps), step=steps)

        if steps % config["save_steps"] == 0:
            manager.save()
            print(f"\n>> saved chekpoint file at {manager.latest_checkpoint}.")

        if steps % config["gen_steps"] == 0:
            image_manager.generate_and_save_images_cnn(model, steps, 2)

    print(f"\n>> training done for {config['net']}!")

    return model
