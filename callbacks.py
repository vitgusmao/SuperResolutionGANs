import tensorflow as tf
import numpy as np

keras = tf.keras

from registry import CALLBACK_REGISTRY


class PerEpochGenerationCallback(keras.callbacks.Callback):
    def __init__(self, per_epoch):
        self.per_epoch = per_epoch
        super().__init__()

    def on_epoch_end(self, epoch, logs=None):

        if epoch % self.per_epoch == 0:
            self.model.img_m.generate_and_save_images(self.model.gen, epoch, 2)


@CALLBACK_REGISTRY.register()
def per_epoch_generation_callback(opts, general_opts=None):
    per_epoch = opts.get("per_epoch")
    return PerEpochGenerationCallback(per_epoch)


@CALLBACK_REGISTRY.register()
def model_checkpoint_callback(opts, general_opts):
    monitor = opts.get("monitor")
    checkpoint_filepath = general_opts.get("checkpoint_path")

    return keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor=monitor,
        mode="max",
        save_best_only=True,
    )
