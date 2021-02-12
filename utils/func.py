import numpy as np


def map_element(func, arr: np.ndarray):
    shape = arr.shape
    flat_arr = arr.flatten()
    return np.array([func(elm) for elm in flat_arr]).reshape(shape)