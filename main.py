from plot.all_informations import plot_togheter
import tensorflow as tf

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

import sys

sys.path.append('./')

def run_net(runner):

    train_args = {'batch_size': 1, 'epochs': 100}

    # Input shapes
    channels = 3

    lr_height = 64
    lr_width = 64
    lr_img_shape = (lr_height, lr_width)
    lr_shape = (lr_height, lr_width, channels)

    hr_height = lr_height * 4
    hr_width = lr_width * 4
    hr_img_shape = (hr_height, hr_width)
    hr_shape = (hr_height, hr_width, channels)

    img_shapes = {
        'hr_shape': hr_shape,
        'lr_shape': lr_shape,
        'hr_img_shape': hr_img_shape,
        'lr_img_shape': lr_img_shape
    }

    dataset_info = {
        'dataset_name': 'DIV2K_train_HR',
        'dataset_dir': '../datasets/{}/',
    }

    runner(img_shapes, dataset_info, train_args)


# from gans.esrgan.evo import train_and_compile
with tf.device('/GPU:0'):
    from nets.vdsr.model import compile_and_train

    run_net(compile_and_train)

# some metrics
# psnr = tf.image.psnr(out, y, max_val=255.0)
# ssim = tf.image.ssim(out, y, max_val=255.0)