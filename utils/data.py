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
        one_hot_matrix = np.zeros(shape=(len(data), len(categories)))
            
        for row_num, category in enumerate(column):
            one_hot_matrix[row_num, categories.index(category)] = 1
        data = np.concatenate((data[:, :column_num], one_hot_matrix, \
            data[:, column_num + 1:]), axis=1)
    return data.astype(np.float64)


def split_train_test(data: np.ndarray, test_ratio=0.1):
    test_size = int(len(data) * test_ratio)
    return data[:-test_size], data[-test_size:]


class DataIterator():
    def __init__(self, data: np.ndarray, output_dim: int=1, \
        batch_size: int=10, drop_remainder: bool=False):
        
        self.data = data
        self.output_dim = output_dim
        self.batch_size = batch_size
        self.drop_remainder = drop_remainder
        self.shuffle_map = np.arange(len(self))

        if not drop_remainder:
            from math import ceil
            self.batch_count = ceil(len(self.data) / self.batch_size)
        else:
            self.batch_count = int(len(self.data) / self.batch_size)

        self.input_data  = self.data[:, :-self.output_dim]
        self.output_data = self.data[:, -self.output_dim:].reshape((-1, 1))
        
        self.xdim = self.input_data.shape[1]
        self.ydim = self.output_data.shape[1]

        self.__iter__()


    def __iter__(self):
        np.random.shuffle(self.shuffle_map)
        shuffle_iter = []
        for batch_num in range(self.batch_count):
            shuffle_iter.append(self.shuffle_map[batch_num * self.batch_size: \
                                                (batch_num + 1) * self.batch_size])
        self.shuffle_iter = iter(shuffle_iter)
        return self


    def __next__(self):
        idx = next(self.shuffle_iter)
        return self.input_data[idx], self.output_data[idx]


    def __len__(self):
        return self.data.shape[0]


    def _reset(self, batch_size):
        self.__init__(self.data, self.output_dim, batch_size, self.drop_remainder)
        return self
