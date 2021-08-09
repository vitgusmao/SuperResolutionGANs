import glob
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import ipdb
import nibabel as nib
from PIL import Image

from data_manager import ImagesManager
from utils import load_yaml
from nets import test_srgan


def run_interpolations(config):

    img_config = config["images"]
    shape = (img_config["gt_size"], img_config["gt_size"])

    methods = {
        "nearest": Image.NEAREST,
        "bilinear": Image.BILINEAR,
        "bicubic": Image.BICUBIC,
        "box": Image.BOX,
        "hamming": Image.HAMMING,
        "lanczos": Image.LANCZOS,
    }

    def get_interpolation(method, shape):
        def interpolate(image):
            image = image.resize(shape, resample=method)

            return image

        return interpolate

    for name, method in methods.items():
        config["name"] = name
        print(f">> generating images with {name}.")
        image_manager = ImagesManager(config)

        interpolation = get_interpolation(method, shape)
        image_manager.test_images_interpolation(interpolation)

    print(f">> test done for interpolations.")


# config = load_yaml("./configs/interpolations.yaml")

# run_interpolations(config)


config = load_yaml("./configs/srgan.yaml")

img_mngr = ImagesManager(config)
net = test_srgan(config)
img_mngr.test_net(net)

print(f">> test done for {config['name']}")

# img_mngr.test_psnr_and_ssim()
