from plot.all_informations import plot_togheter
import tensorflow as tf

keras = tf.keras
gpu_devices = tf.config.experimental.list_physical_devices("GPU")
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

import sys

sys.path.append("./")

from registry import MODEL_REGISTRY, CALLBACK_REGISTRY
from callbacks import *
from data_manager import ImagesManager
from nets.srgan import model
from nets.esrgan import model

# CALLBACK_REGISTRY.register(per_epoch_generation_callback)
# CALLBACK_REGISTRY.register(model_checkpoint_callback)

opts = {
    "net": "esrgan",
    "images": {
        "lr_size": 64,
        "hr_size": 256,
        "channels": 3,
    },
    "datasets": {
        "train": {
            "name": "img_align_celeba",
            "dir": "../datasets/img_align_celeba/",
        },
        "test": {
            "name": "img_align_celeba",
            "dir": "../datasets/img_align_celeba/",
            "size": 10,
        },
    },
    # train
    "train": {
        "generator": {
            "lr": 1e-5
        },
        "discriminator": {
            "lr": 4e-5
        }
    },
    "epochs": 10,
    "batch_size": 1,
    "callbacks": {
        "per_epoch_generation_callback": {
            "per_epoch": 1,
        },
        "model_checkpoint_callback": {
            "monitor": "psnr",
        },
    },
    "checkpoint_path": "checkpoints/esrgan",
}


# from gans.esrgan.evo import train_and_compile
with tf.device("/GPU:0"):
    callbacks = opts.get("callbacks")
    compiled_callbacks = []
    for method, items in callbacks.items():
        callback = CALLBACK_REGISTRY.get(method)
        compiled_callbacks.append(callback(items, opts))

    build_net = MODEL_REGISTRY.get(opts.get("net"))

    batch_size = opts.get("batch_size")
    epochs = opts.get("epochs")

    image_manager = ImagesManager(opts)

    image_loader = image_manager.get_dataset()
    image_manager.initialize_dirs(2, epochs)

    net = build_net(opts, image_manager)

    try:
        net.load_weights(opts.get("checkpoint_path"))
    except Exception:
        pass

    history = net.fit(
        image_loader,
        batch_size=batch_size,
        epochs=epochs,
        #  use_multiprocessing=True,
        #  workers=2,
        callbacks=compiled_callbacks,
    )
