from plot.all_informations import plot_togheter
import tensorflow as tf

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

from plot.losses import plot_losses
import sys

from ESRGAN.esrgan import build_esrgan_net

sys.path.append('./')

esrgan = build_esrgan_net()

information = esrgan(epochs=300, batch_size=1, sample_interval=20)

plot_togheter(information,
              ['d_loss', 'g_loss', 'd_fake_loss', 'd_real_loss', 'valid_g'])
