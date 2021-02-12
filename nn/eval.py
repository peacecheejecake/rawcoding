import nn.loss as loss

import numpy as np


def error_ratio_batch_sum(batch_pred, batch_out):
    error_ratio = np.abs(batch_pred - batch_out) / batch_out
    return np.sum(error_ratio)


def exact_correct_batch_count(batch_pred, batch_out):
    is_correct = np.equal(batch_pred, batch_out)
    return np.sum(is_correct)