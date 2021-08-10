import glob
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import ipdb
import nibabel as nib
from PIL import Image
import argparse

from data_manager import ImagesManager
from utils import load_yaml
from nets import test_srgan

parser = argparse.ArgumentParser(description="Options for super resolution")
parser.add_argument(
    "--config", help="name of yaml config file under configs/", default=None
)

args = parser.parse_args()


def run_interpolations(config):

    img_config = config["images"]
    shape = (img_config["gt_size"], img_config["gt_size"])

    methods = {
        "nearest": Image.NEAREST,
        "bilinear": Image.BILINEAR,
        "bicubic": Image.BICUBIC,
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

if args.config:
    config = load_yaml(f"./configs/{args.config}.yaml")
    print(f">> {config['name']} config file loaded")
    img_mngr = ImagesManager(config)
    net = test_srgan(config)
    img_mngr.test_net(net)
    print(f">> {config['name']} results printed.")
else:
    raise Exception(">> missing config file.")


# img_mngr.test_psnr_and_ssim()
