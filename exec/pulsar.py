from nn.models import SingleLayerPerceptron
import os.path

import numpy as np

from path import *

import utils.data
from utils.data import DataIterator
from nn.activ import sigmoid
from nn.eval import exact_correct_batch_count
import nn.loss


def main(lr=1e-3, epochs=10, batch_size=10, test_ratio=0.1, test_size=50, \
        title_width=40, title_fillchar='=', report_every=1):

    # arrange data
    path_to_data = ['..', 'data', 'pulsar']
    filenames = ['pulsar_data_train.csv', 'pulsar_data_test.csv']
    train_data = utils.data.read_csv(os.path.join(path_to_data, filenames[0]))
    test_data = utils.data.read_csv(os.path.join(path_to_data, filenames[1]))

    train_iter = DataIterator(data=train_data, batch_size=batch_size)
    test_iter = DataIterator(data=test_data, batch_size=batch_size)

    # create a model and initialize parameters
    loss_out = nn.loss.CrossEntropy()
    PulsarClassifier = SingleLayerPerceptron(name='pulsar', \
        xdim=train_iter.xdim, ydim=train_iter.ydim, \
        activ_func=sigmoid, loss=loss_out, \
        eval_func=exact_correct_batch_count)
    PulsarClassifier.init()

    # first evaluation before training
    print(" Initial Evaluation Accuracy ".center(title_width, '='))
    print(f"Train Accuracy: [{PulsarClassifier.evaluate(train_iter)}], \
Test Accuracy: [{PulsarClassifier.evaluate(test_iter)}]", end='\n\n')

    # train
    print(" Start of Training ".center(title_width, '='))
    for epoch in range(1, epochs + 1):
        loss_epoch_sum = 0
        for batch_in, batch_out in train_iter:
            loss_epoch_sum += PulsarClassifier.backward(batch_in, batch_out, lr)
        loss = loss_epoch_sum / len(train_data)

        train_accuracy = PulsarClassifier