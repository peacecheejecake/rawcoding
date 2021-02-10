import numpy as np


class Module():
    def __init__(self, name='module'):
        pass



class Linear():
    def __init__(self, name='lin', xdim=10, ydim=1):
        self.name = name
        self.xdim = xdim
        self.ydim = ydim


    def init(self):
        self.weight = np.random.normal(size=(self.xdim, self.ydim))
        self.bias = 0