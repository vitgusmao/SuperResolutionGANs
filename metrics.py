import ipdb
import tensorflow as tf
import numpy as np

keras = tf.keras

from utils import denormalize
from registry import METRIC_REGISTRY


@METRIC_REGISTRY.register()
def psnr(y_true, y_pred):
    y_pred = denormalize(y_pred)
    y_true = denormalize(y_true)
    return tf.image.psnr(y_pred, y_true, max_val=255)


@METRIC_REGISTRY.register()
def ssim(y_true, y_pred):
    y_pred = denormalize(y_pred)
    y_true = denormalize(y_true)
    return tf.image.ssim(y_pred, y_true, max_val=255)


@METRIC_REGISTRY.register()
def accuracy(y_true, y_pred):
    metric = tf.keras.metrics.BinaryCrossentropy()
    return metric(y_true, y_pred)
