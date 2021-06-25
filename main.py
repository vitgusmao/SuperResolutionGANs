from plot.all_informations import plot_togheter
import tensorflow as tf

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

import sys

sys.path.append('./')

from gans.esrgan.esrgan import build_esrgan_net as build_gan
# from gans.srgan.srgan import build_srgan_net as build_gan

gan = build_gan()

information = gan(
    epochs=200,
    batch_size=1,
    sample_interval=20,
)

plot_togheter(information)