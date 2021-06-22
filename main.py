from plot.losses import plot_losses
import sys

from ESRGAN.esrgan import build_esrgan_net

sys.path.append('./')

esrgan = build_esrgan_net()

losses = esrgan(epochs=1000, batch_size=1, sample_interval=20)

plot_losses(losses)