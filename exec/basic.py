
import abc

import numpy as np

from utils.data import DataIterator
import utils.data
from nn.basic import Module
from nn.loss import Loss

from typing import TypeVar, Any


P = TypeVar('P', str, list)


class Executer(metaclass=abc.ABCMeta):
    def __init__(self, train_iter: DataIterator=None, test_iter: DataIterator=None, \
        model: Module=None, loss: Loss=None, title_width: int=40, title_fillchar: str='='):
        self.train_iter = train_iter
        self.test_iter = test_iter
        self.model = model
        self.loss = loss
        self.title_width = title_width
        self.title_fillchar = title_fillchar
    

    def arrange_data(self, test_ratio: float, batch_size: int):
        pass

    
    def construct_model(self):
        pass
    

    def first_eval(self):
        print(" Initial Evaluation Accuracy ".center(self.title_width, self.title_fillchar))
        print(f"Train Accuracy: [{self.model.evaluate(self.train_iter)}], \
Test Accuracy: [{self.model.evaluate(self.test_iter)}]", end='\n\n')


    def train_model(self, epochs: int=10, lr: float=1e-3, report_every=1):
        print(" Start of Training ".center(self.title_width, self.title_fillchar))
        for epoch in range(1, epochs + 1):
            loss_epoch_sum = 0
            for batch_in, batch_out in self.train_iter:
                loss_epoch_sum += self.model.backward(batch_in, batch_out, lr)
            loss_out = loss_epoch_sum / len(self.train_iter)

            train_accuracy = self.model.evaluate(self.train_iter)
            test_accuracy = self.model.evaluate(self.test_iter)
            if epoch % report_every == 0 and epoch != epochs:
                print(f"Epoch[{str(epoch).rjust(2)}] --- Loss: [{loss_out:.3f}],", \
                    f"Train Accuracy: [{train_accuracy:.3f}]," \
                    f"Test Accuracy: [{test_accuracy:.3f}]")

        print("End of Training.", end='\n\n')


    def show_sample_result(self, cal_size: int=50, show_size: int=50, label_width: int=2, label_dtype=None):
        print(f" Sample Test Result({show_size}) ".center(self.title_width, self.title_fillchar))

        batch_in, batch_out = next(self.test_iter._reset(cal_size))
        predict_out = self.model.forward(batch_in)

        if label_dtype:
            predict_out = predict_out.astype(label_dtype)
            batch_out = batch_out.astype(label_dtype)
        
        correct_count = 0
        for sample_num, (predict, answer) in enumerate(zip(predict_out, batch_out)):
            if predict == answer:
                is_correct = "Correct!"
                correct_count += 1
            else:
                is_correct = "Wrong!"

            if sample_num > show_size:
                continue

            print(f"{is_correct.ljust(8)}   Predicted: {str(predict.item()).rjust(label_width)}, \
    Answer: {str(answer.item()).rjust(label_width)}")

        print(f"> Correct Ratio:  [{correct_count / cal_size:.5f}]", end='\n\n')


    def final_eval(self):
        print(" Final Test ".center(self.title_width, self.title_fillchar))
        print(f"Train Accuracy: [{self.model.evaluate(self.train_iter):.5f}]")
        print(f"Test Accuracy:  [{self.model.evaluate(self.test_iter):.5f}]", end='\n\n')


    def check_parameters(self):
        print(" Parameters ".center(self.title_width, self.title_fillchar))
        # print(f"Weight: {AbaloneRegressor.linear.weight.reshape(-1)}")
        # print(f"Bias: {AbaloneRegressor.linear.bias.reshape(-1)}")
        for param_name, param_value in self.model.parameters.items():
            print(f"{param_name.title()}: {param_value}")
        print()