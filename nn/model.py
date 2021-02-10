import numpy as np

class Linear():
    def __init__(self, name='lin', xdim=10, ydim=1):
        self.name = name
        self.xdim = xdim
        self.ydim = ydim


    def init(self):
        self.weight = np.random.normal(size=(self.xdim, self.ydim))
        self.bias = 0


class SingleLayerPerceptron():
    '''
    Single Layer Perceptron Class
    '''

    def __init__(self, name='slp', xdim=10, ydim=1):
        self.name = name
        self.xdim = xdim
        self.ydim = ydim
        self.linear = Linear(name='slp', xdim=self.xdim, ydim=self.ydim)
        self.init_parameters()


    def init_parameters(self):
        self.linear.init()
    

    def forward(self, x):
        net = np.matmul(x, self.linear.weight) + self.linear.bias
        return net


    def evaluate(self, data_iter):
        num_correct = 0
        # sum_squared_error = 0
        sum_error_ratio = 0
        for batch_in, batch_out in data_iter:
            predict_out = self.forward(batch_in)
            # compare_out = (predict_out == batch_out)
            # num_correct = np.sum(compare_out)
            error_ratio = np.abs(predict_out - batch_out) / batch_out
            sum_error_ratio += np.sum(error_ratio)

        # accuracy = num_correct / len(data_iter)
        accuracy = 1 - sum_error_ratio / len(data_iter)
        
        return accuracy


    def backward(self, batch_in, batch_out, learning_rate):
        errors = batch_out - self.forward(batch_in)
        norm = np.linalg.norm(errors)
        batch_size = len(batch_out)
        loss = norm / len(batch_in) / batch_size
        
        g_loss_by_output = -errors / norm                                    # n * 1
        g_output_by_weight = batch_in.T                                      # m * n
        g_output_by_bias = np.ones(shape=(1, batch_size))                    # 1 * n

        gradient_weight = np.matmul(g_output_by_weight, g_loss_by_output)    # m * 1
        gradient_bias = np.matmul(g_output_by_bias, g_loss_by_output)        # 1 * 1

        self.linear.weight -= learning_rate * gradient_weight
        self.linear.bias -= learning_rate * gradient_bias
        
        return loss
