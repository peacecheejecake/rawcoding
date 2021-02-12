import numpy as np


class Module():
    def __init__(self, name='module'):
        pass


class Linear(Module):
    def __init__(self, xdim: int, ydim: int, name: str='linear'):
        self.name = name
        self.xdim = xdim
        self.ydim = ydim


    @property
    def parameters(self):
        return {'weight': self.weight.T, 'bias': self.bias}


    def init(self, init_func=None):
        if not init_func:
            self.weight = np.random.normal(size=(self.xdim, self.ydim))
            self.bias = 0
        else:
            self.weight, self.bias = init_func(self.xdim, self.ydim)
