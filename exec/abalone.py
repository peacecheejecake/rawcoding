import os.path
import numpy as np

from path import *

import utils.data
from utils.data import DataIterator

from nn.models import SingleLayerPerceptron
from nn.loss import Loss, RMSE
from nn.basic import Module
import nn.eval

from basic import Executer


class AbaloneExecuter(Executer):
    def __init__(self, train_iter: DataIterator=None, test_iter: DataIterator=None, \
        model: Module=None, loss: Loss=None, title_width: int=40, title_fillchar: str='='):
        self.train_iter = train_iter
        self.test_iter = test_iter
        self.model = model
        self.loss = loss
        self.title_width = title_width
        self.title_fillchar = title_fillchar


    def arrange_data(self, data_path: str, test_ratio: float=0.1, batch_size: int=10, output_cols: int=1):
        raw_data = utils.data.read_csv(data_path, False)
        data = utils.data.one_hot(raw_data, [0]).astype(np.float64)
        train_data, test_data = utils.data.split_train_test(data=data, test_ratio=test_ratio)

        self.train_iter = DataIterator(data=train_data, output_dim=output_cols, \
            batch_size=batch_size, drop_remainder=False)
        self.test_iter  = DataIterator(data=train_data, output_dim=output_cols, \
            batch_size=batch_size, drop_remainder=False)


    def construct_model(self, model_name='abalone_regressor'):
        self.loss = RMSE()
        self.model = SingleLayerPerceptron(name=model_name, \
            xdim=self.train_iter.xdim, ydim=self.train_iter.ydim, \
            loss=self.loss, activ_func=None, eval_func=nn.eval.error_ratio_batch_sum)
        self.model.init()


def main():
    # make an executer and start program
    executer = AbaloneExecuter()

    # data arrangement
    data_path = os.path.join('..', 'data', 'abalone.csv')
    executer.arrange_data(data_path=data_path)
    
    executer.construct_model()                                          # create a model and initialize
    executer.first_eval()                                               # first evaluation before training
    executer.train_model(epochs=100, lr=1e-3, report_every=10)          # train model

    # final test
    executer.show_sample_result(cal_size=len(executer.test_iter), label_dtype=np.int64)  # sample results
    executer.final_eval()          # accuracy

    executer.check_parameters()    # check parameters


if __name__ == '__main__':
    main()