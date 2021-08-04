import tensorflow as tf
import numpy as np

from registry import MODEL_REGISTRY
from utils import ProgressBar
from data_manager import ImagesManager, load_datasets, define_image_process

from metrics import psnr, ssim
from nets.srgan.generator import RB_Model
from nets.srgan.discriminator import Discriminator
from lr_schedule import MultiStepLR
from losses import GeneratorLoss, DiscriminatorLoss, ContentLoss


@MODEL_REGISTRY.register()
def srgan(config):
    
    image_manager = ImagesManager(config)
    image_manager.initialize_dirs(2, config["epochs"])

    imgs_config = config["images"]

    train_config = config["train"]
    g_config = train_config["generator"]
    d_config = train_config["discriminator"]

    # define network
    generator = RB_Model(imgs_config["gt_size"], imgs_config["scale"], imgs_config["channels"])
    discriminator = Discriminator(imgs_config["gt_size"], imgs_config["channels"])

    # load dataset
    train_dataset = load_datasets(
        config["datasets"], "train_datasets", config["batch_size"], shuffle=False
    )
    # image_loader = image_manager.get_dataset()
    process_image = define_image_process(imgs_config["gt_size"], imgs_config["scale"])

    # define losses function
    fea_loss_fn = ContentLoss(criterion="l1")
    gen_loss_fn = GeneratorLoss(gan_type="ragan")
    dis_loss_fn = DiscriminatorLoss(gan_type="ragan")

    # define optimizer
    learning_rate_G = MultiStepLR(
        g_config["lr"], train_config["lr_steps"], train_config["lr_rate"]
    )
    learning_rate_D = MultiStepLR(
        d_config["lr"], train_config["lr_steps"], train_config["lr_rate"]
    )
    optimizer_G = tf.keras.optimizers.Adam(
        learning_rate=learning_rate_G,
        beta_1=g_config["adam_beta1"],
        beta_2=g_config["adam_beta2"],
    )
    optimizer_D = tf.keras.optimizers.Adam(
        learning_rate=learning_rate_D,
        beta_1=d_config["adam_beta1"],
        beta_2=d_config["adam_beta2"],
    )

    model_ema = tf.train.ExponentialMovingAverage(decay=train_config["ema_decay"])

    # load checkpoint
    checkpoint_dir = "./checkpoints/" + config["net"]
    checkpoint = tf.train.Checkpoint(
        step=tf.Variable(0, name="step"),
        optimizer_G=optimizer_G,
        optimizer_D=optimizer_D,
        model=generator,
        discriminator=discriminator,
    )
    manager = tf.train.CheckpointManager(
        checkpoint=checkpoint, directory=checkpoint_dir, max_to_keep=3
    )

    if manager.latest_checkpoint:
        checkpoint.restore(manager.latest_checkpoint)
        print(
            f">> load ckpt from {manager.latest_checkpoint} at step {checkpoint.step.numpy()}."
        )
    else:
        print(">> training from scratch.")

    # define training step function
    @tf.function
    def train_step(lr, hr):
        step_output = {}

        with tf.GradientTape(persistent=True) as tape:
            sr = generator(lr, training=True)
            hr_output = discriminator(hr, training=True)
            sr_output = discriminator(sr, training=True)

            losses_G = {}
            losses_D = {}
            losses_G["reg"] = tf.reduce_sum(generator.losses)
            losses_D["reg"] = tf.reduce_sum(discriminator.losses)
            losses_G["feature"] = train_config["feature_weight"] * fea_loss_fn(hr, sr)
            losses_G["gan"] = train_config["gen_weight"] * gen_loss_fn(
                hr_output, sr_output
            )
            losses_D["gan"] = dis_loss_fn(hr_output, sr_output)
            total_loss_G = tf.add_n([l for l in losses_G.values()])
            total_loss_D = tf.add_n([l for l in losses_D.values()])

        grads_G = tape.gradient(total_loss_G, generator.trainable_variables)
        grads_D = tape.gradient(total_loss_D, discriminator.trainable_variables)

        optimizer_G.apply_gradients(zip(grads_G, generator.trainable_variables))
        optimizer_D.apply_gradients(zip(grads_D, discriminator.trainable_variables))

        # with tf.control_dependencies([gen_op]):
        #     self.ema.apply(generator.trainable_variables)
        # with tf.control_dependencies([disc_op]):
        #     self.ema.apply(discriminator.trainable_variables)

        # step_output.update({m.__name__: m(hr, sr) for m in gen_metrics})
        # step_output.update(
        #     {
        #         f"{m.__name__}_real": m(tf.ones_like(hr_output), hr_output)
        #         for m in disc_metrics
        #     }
        # )
        # step_output.update(
        #     {
        #         f"{m.__name__}_fake": m(tf.zeros_like(sr_output), sr_output)
        #         for m in disc_metrics
        #     }
        # )

        # step_output.update(
        #     {
        #         "loss_G": total_loss_G,
        #         "loss_D": total_loss_D,
        #     }
        # )

        return total_loss_G, total_loss_D, losses_G, losses_D

    # training loop
    summary_writer = tf.summary.create_file_writer("./logs/" + config["net"])
    prog_bar = ProgressBar(config["epochs"], checkpoint.step.numpy())
    remain_steps = max(config["epochs"] - checkpoint.step.numpy(), 0)

    for raw_img in train_dataset.take(remain_steps):
        lr, hr = process_image(raw_img)

        checkpoint.step.assign_add(1)
        steps = checkpoint.step.numpy()

        total_loss_G, total_loss_D, losses_G, losses_D = train_step(lr, hr)

        prog_bar.update(
            "loss_G={:.4f}, loss_D={:.4f}, lr_G={:.1e}, lr_D={:.1e}".format(
                total_loss_G.numpy(),
                total_loss_D.numpy(),
                optimizer_G.lr(steps).numpy(),
                optimizer_D.lr(steps).numpy(),
            )
        )

        if steps % 10 == 0:
            with summary_writer.as_default():
                tf.summary.scalar("loss_G/total_loss", total_loss_G, step=steps)
                tf.summary.scalar("loss_D/total_loss", total_loss_D, step=steps)
                for k, l in losses_G.items():
                    tf.summary.scalar("loss_G/{}".format(k), l, step=steps)
                for k, l in losses_D.items():
                    tf.summary.scalar("loss_D/{}".format(k), l, step=steps)

                tf.summary.scalar("learning_rate_G", optimizer_G.lr(steps), step=steps)
                tf.summary.scalar("learning_rate_D", optimizer_D.lr(steps), step=steps)

        if steps % config["save_steps"] == 0:
            manager.save()
            print(f"\n>> saved chekpoint file at {manager.latest_checkpoint}.")

        if steps % config["gen_steps"] == 0:
            image_manager.generate_and_save_images(generator, steps, 2)

    print(f"\n>> {config['net']} training done!")
