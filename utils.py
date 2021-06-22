import numpy as np


def normalize(
    input_data,
    new_max=1,
    new_min=0,
    max_value=255,
    min_value=0,
):

    scale = new_max - new_min

    return (scale * ((input_data.astype(np.float32) - min_value) /
                     (max_value - min_value))) + new_min


def denormalize(
    input_data,
    new_max=255,
    new_min=0,
    max_value=1,
    min_value=0,
):
    """
    Args:
        input_data (np.array): Imagem normalizada em formato de array

    Returns:
        np.array: Imagem desnormalizada com pixels de [0 - 255] no formato uint8
    """
    scale = max_value - min_value

    input_data = ((input_data - min_value) / scale) * (new_max - new_min)

    if (np.amax(input_data) > 255) and (np.amin(input_data) < 0):
        raise ValueError

    return input_data.astype(np.uint8)
