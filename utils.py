import numpy as np


def cant_finish_with_bar(path):
    if path[-1] == "/":
        path.pop(-1)
    return path


def must_finish_with_bar(path):
    if path[-1] != "/":
        path += "/"
    return path


def check_pixels(image, max=255, min=0):
    if (np.amax(image) > max) and (np.amin(image) < min):
        raise Exception(
            "Valor do pixels utrapassou o limite do intervalo [0, 255]."
            + f"Valor mínimo encontrado {np.amin(image)}, valor máximo encontrado {np.amax(image)}"
        )


def normalize(image):
    new_min, new_max = -1, 1
    min_value, max_value = 0, 255
    scale = new_max - new_min

    output = (scale * ((image - min_value) / (max_value - min_value))) + new_min
    check_pixels(output, max=1, min=-1)

    return output


def denormalize(image):
    """
    Args:
        image (np.array): Imagem normalizada em formato de array

    Returns:
        np.array: Imagem desnormalizada com pixels de [0 - 255] no formato uint8
    """
    min_value, max_value = -1, 1
    new_min, new_max = 0, 255
    scale = max_value - min_value

    output = ((image - min_value) / scale) * (new_max - new_min)
    check_pixels(output)

    return output
