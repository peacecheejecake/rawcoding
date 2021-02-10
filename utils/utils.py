import csv
import os.path

import numpy as np


def read_csv(filepath: list, filename: str, include_headers: bool=False):
    with open(os.path.join(*filepath, filename)) as f:
        csv_reader = csv.reader(f)
        if not include_headers:
            next(csv_reader)
        data = np.array(list(csv_reader))
    return data


def one_hot(data: np.ndarray, column_nums: list):
    data = data.copy()
    for column_num in column_nums:
        column = data[:, column_num]
        categories = list(set(column))
        one_hot_matrix = np.zeros(shape=(len(data), len(categories)), \
            dtype=np.int64)
            
        for row_num, category in enumerate(column):
            one_hot_matrix[row_num, categories.index(category)] = 1
        data = np.concatenate((data[:, :column_num], one_hot_matrix, \
            data[:, col_num + 1:]), axis=1)
    return data