import numpy as np


class Loss():
    def __init__(self, predict_out=None, batch_out=None, batch_in=None):
        self.predict_out = predict_out
        self.batch_out = batch_out
        self.batch_in = batch_in
        if batch_in:
            self.batch_size = len(batch_in)


    def __call__(self, predict_out, batch_out, batch_in):
        self.predict_out = predict_out
        self.batch_out = batch_out
        self.batch_in = batch_in
        self.batch_size = len(batch_in)


class RMSE(Loss):
    def __init__(self, predict_out=None, batch_out=None, batch_in=None):
        super().__init__(predict_out, batch_out, batch_in)


    def __call__(self, predict_out, batch_out, batch_in):
        super().__call__(predict_out, batch_out, batch_in)
        return self.forward(predict_out, batch_out)


    def forward(self, predict_out, batch_out):
        self.error = predict_out - batch_out
        self.norm_l2 = np.linalg.norm(self.error)
        return self.norm_l2 / len(batch_out)


    def backward(self):
        d_loss_by_output = self.error / self.norm_l2
        d_output_by_weight = self.batch_in.T
        d_output_by_bias = np.ones(shape=(1, self.batch_size))

        g_weight = np.matmul(d_output_by_weight, d_loss_by_output)
        g_bias = np.matmul(d_output_by_bias, d_loss_by_output)

        return g_weight, g_bias


class CrossEntropy(Loss):
    def __init__(self, predict_out=None, batch_out=None, batch_in=None,\
        epsilon=1e-10):
        super().__init__(predict_out, batch_out, batch_in)
        self.epsilon = epsilon

        if predict_out:
            self.q = self.make_q(predict_out)
        if batch_out:
            self.p = batch_out


    def __call__(self, predict_out, batch_out, batch_in):
        super().__call__(predict_out, batch_out, batch_in)
        self.p = self.batch_out
        self.q = self.make_q()
        
        return self.forward()


    def forward(self):
        return np.sum(-self.p * np.log(self.q))


    def backward(self):
        d_entropy_over_logit = self.q - self.p
        d_logit_over_weight = self.batch_in.T
        d_logit_over_bias = np.ones(shape=self.batch_in.shape).T

        g_weight = np.matmul(d_logit_over_weight, d_entropy_over_logit)
        g_bias = np.matmul(d_logit_over_bias, d_entropy_over_logit)

        return g_weight, g_bias


    def make_q(self):
        if self.predict_out.shpae[1] == 1:
            return np.hstack((self.predict_out, 1 - self.predict_out))
        else:
            return self.predict_out