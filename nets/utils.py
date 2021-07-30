import numpy as np

DIV2K_RGB_MEAN = np.array([0.4488, 0.4371, 0.4040]) * 255


def normalize(x, rgb_mean=DIV2K_RGB_MEAN):
    '''This function will normalize image by substracting RGB mean from image'''
    return (x - rgb_mean) / 127.5


def denormalize(x, rgb_mean=DIV2K_RGB_MEAN):
    ''' This function will denormalize image by adding back rgb_mean'''
    return (x * 127.5) + rgb_mean