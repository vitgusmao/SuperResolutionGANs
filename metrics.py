import ipdb
import numpy as np
import tensorflow as tf


def psnr(y_true, y_pred):
    tf.image.psnr(y_pred, y_true, max_val=255)


def ssim(y_true, y_pred):
    tf.image.ssim(y_pred, y_true, max_val=255)


def image_metrics():
    return [{'method': psnr, 'name': 'psnr'}, {'method': ssim, 'name': 'ssim'}]

