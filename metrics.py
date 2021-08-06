import ipdb
import tensorflow as tf
import numpy as np

from utils import denormalize
from registry import METRIC_REGISTRY

def rgb2ycbcr(img, only_y=True):
    """Convert rgb to ycbcr
    only_y: only return Y channel
    Input:
        uint8, [0, 255]
        float, [0, 1]
    """
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    img = img[:, :, ::-1]

    # convert
    if only_y:
        rlt = np.dot(img, [24.966, 128.553, 65.481]) / 255.0 + 16.0
    else:
        rlt = np.matmul(
            img, [[24.966, 112.0, -18.214],
                  [128.553, -74.203, -93.786],
                  [65.481, -37.797, 112.0]]) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)

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


# calculate_psnr(rgb2ycbcr(bic_img), rgb2ycbcr(hr_img)),
# calculate_ssim(rgb2ycbcr(bic_img), rgb2ycbcr(hr_img)),


