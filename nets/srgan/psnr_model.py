import tensorflow as tf

from utils import ProgressBar, load_history, save_history
from registry import MODEL_REGISTRY
from data_manager import define_image_process, load_datasets, ImagesManager

from metrics import psnr, ssim
from .generator import RB_Model
from lr_schedule import MultiStepLR
from losses import PixelLoss


@MODEL_REGISTRY.register()
def gan_pretrain(config):

    image_manager = ImagesManager(config)
    image_manager.initialize_dirs(2, config["epochs"])

    history = load_history(config)

    imgs_config = config["images"]
    train_config = config["train"]

    # define network
    model = RB_Model(
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

            total_loss = train_config["pixel_weight"] * pixel_loss_fn(hr, sr)

        grads = tape.gradient(total_loss, checkpoint.model.trainable_variables)
        checkpoint.optimizer.apply_gradients(
            zip(grads, checkpoint.model.trainable_variables)
        )

        return total_loss, sr

    # training loop
    prog_bar = ProgressBar(config["epochs"], checkpoint.step.numpy())
    remain_steps = max(config["epochs"] - checkpoint.step.numpy(), 0)

    for raw_img in train_dataset.take(remain_steps):
        lr, hr = process_image(raw_img)

        checkpoint.step.assign_add(1)
        steps = checkpoint.step.numpy()

        total_loss, sr = train_step(lr, hr)

        prog_bar.update(
            "loss={:.4f}, lr={:.1e}".format(
                total_loss.numpy(), optimizer.lr(steps).numpy()
            )
        )

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
            image_manager.generate_and_save_images(model, steps, 2)

    print(f"\n>> training done for {config['name']}!")

    return model
