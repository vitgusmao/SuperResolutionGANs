from tensorflow._api.v2 import config
from plot.all_informations import plot_togheter
import tensorflow as tf

gpu_devices = tf.config.experimental.list_physical_devices("GPU")
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

import sys
import yaml

sys.path.append("./")


from registry import MODEL_REGISTRY
from utils import load_yaml
from data_manager import ImagesManager
from nets.srgan import model
from nets.esrgan import model
from nets.esrgan import psnr_model
from nets.srcnn import model

# from nets.vdsr import model


with tf.device("/GPU:0"):

    config = load_yaml("./configs/pretrain.yaml")

    build_net = MODEL_REGISTRY.get(config.get("net"))

    build_net(config)
