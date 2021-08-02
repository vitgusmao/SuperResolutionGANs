import ipdb
import tensorflow as tf
import numpy as np

keras = tf.keras

from registry import METRIC_REGISTRY


def denormalize(input_data, scale_map=(0, 1)):
    """
    Args:
        input_data (np.array): Imagem normalizada em formato de array

    Returns:
        np.array: Imagem desnormalizada com pixels de [0 - 255] no formato uint8
    """
    min_value, max_value = scale_map
    new_max = 255
    new_min = 0
    scale = max_value - min_value

    input_data = ((input_data - min_value) / scale) * (new_max - new_min)

    if (np.amax(input_data) > 255) and (np.amin(input_data) < 0):
        raise Exception(
            f"Valor do pixels utrapassou o limite do intervalo [0, 255]. Valor mínimo encontrado {np.amin(input_data)}, valor máximo encontrado {np.amax(input_data)}"
        )

    return input_data


@METRIC_REGISTRY.register()
def psnr(y_true, y_pred):
    y_pred = denormalize(y_pred, (-1, 1))
    y_true = denormalize(y_true, (-1, 1))
    return tf.image.psnr(y_pred, y_true, max_val=255)


@METRIC_REGISTRY.register()
def ssim(y_true, y_pred):
    y_pred = denormalize(y_pred, (-1, 1))
    y_true = denormalize(y_true, (-1, 1))
    return tf.image.ssim(y_pred, y_true, max_val=255)


@METRIC_REGISTRY.register()
def accuracy(y_true, y_pred):
    metric = tf.keras.metrics.BinaryCrossentropy()
    return metric(y_true, y_pred)
