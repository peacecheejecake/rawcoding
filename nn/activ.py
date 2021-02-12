import numpy as np


def sigmoid(x):
    return np.exp(-relu(-x)) / (1 + np.exp(-abs(x)))


def relu(x):
    return max(x, 0)