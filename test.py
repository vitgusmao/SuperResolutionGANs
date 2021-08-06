import glob
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import ipdb
import nibabel as nib
from PIL import Image

from data_manager import ImagesManager
from utils import load_yaml


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
        image_manager = ImagesManager(config)
        tester = image_manager.test_images_interpolation()

        interpolation = get_interpolation(method, shape)
        tester(interpolation)


# dataset_name = "DIV2K_train_HR"
# dataset_dir = "../datasets/{}/"
# image_manager = ImagesManager(
#     dataset_dir, dataset_name, "None", hr_img_shape, lr_img_shape
# )


# images = image_manager.rebuild_images(images, generated=False)
# for i, img in enumerate(images):
#     img.save(f"{i}_i.png")

# other = image_manager.rebuild_images(other, generated=False)
# for i, img in enumerate(other):
#     img.save(f"{i}_o.png")
config = load_yaml("./configs/interpolations.yaml")

run_interpolations(config)
