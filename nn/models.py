import numpy as np

from nn.basic import Module, Linear
import nn.loss
import utils.func as F

from typing import Callable


class SingleLayerPerceptron(Module):
    '''
    Single Layer Perceptron Class
    '''

    def __init__(self, name: str, xdim: int, ydim: int, loss: nn.loss.Loss, \
        eval_func: Callable[[np.ndarray, np.ndarray], float], \
        activ_func: Callable=None):
        self.name = name
        self.xdim = xdim
        self.ydim = ydim
        self.activ_func = activ_func
        self.loss = loss
        self.eval_func = eval_func

        self._parameters = dict()


    @property
    def parameters(self):
        self._parameters.update(self.linear.parameters)
        return self._parameters


    def init(self, layer_name='linear') -> None:
        self.linear = Linear(xdim=self.xdim, ydim=self.ydim, name=layer_name)
        self.linear.init()
    

    def forward(self, x) -> np.ndarray:
        net = np.matmul(x, self.linear.weight) + self.linear.bias
        if self.activ_func:
            net = F.map_element(self.activ_func, net)
        return net


    def evaluate(self, data_iter, eval_func=None) -> float:
        sum_ = 0
        for batch_in, batch_out in data_iter:
            predict_out = self.forward(batch_in)
            if not eval_func:
                sum_ += self.eval_func(predict_out, batch_out)
            else:
                sum_ += eval_func(predict_out, batch_out)

        accuracy = 1 - sum_ / len(data_iter)
        return accuracy


    def backward(self, batch_in, batch_out, learning_rate) -> float:
        prediction = self.forward(batch_in)
        batch_loss = self.loss(prediction, batch_out, batch_in)
        gradients = self.loss.backward()

        self.linear.weight -= learning_rate * gradients[0]
        self.linear.bias -= learning_rate * gradients[1]
        
        return batch_loss

