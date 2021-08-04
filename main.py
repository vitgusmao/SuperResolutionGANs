import sys
import tensorflow as tf
import argparse
gpu_devices = tf.config.experimental.list_physical_devices("GPU")
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)


sys.path.append("./")


from registry import MODEL_REGISTRY
from utils import load_yaml
from data_manager import ImagesManager
from nets.srgan import model
from nets.esrgan import model
from nets.esrgan import psnr_model
from nets.srcnn import model
from nets.vdsr import model
from nets.edsr import model

from plot.all_informations import plot_togheter


parser = argparse.ArgumentParser(description="Options for super resolution")
parser.add_argument("--config", help="path to yaml config file", default="./configs/pretrain.yaml")

args = parser.parse_args()

with tf.device("/GPU:0"):
    config = load_yaml(args.config)
    build_net = MODEL_REGISTRY.get(config.get("net"))
    build_net(config)
