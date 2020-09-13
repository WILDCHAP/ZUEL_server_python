import numpy as np
from function_1 import softmax, cross_entropy_error

class Affine():
    def __init__(self, w, b):
        self.w = w
        self.b = b
        self.x = None
        self.dw = None
        self.db = None
    def forward(self, x):
        self.x = x
        return np.dot(self.x, self.w) + self.b
    def backward(self, dout):
        self.dw = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        return np.dot(dout, self.w.T)

class Sigmoid():
    def __init__(self):
        self.y = None
    def forward(self, x):
        self.y = 1 / (1+np.exp(-x))
        return self.y
    def backward(self, dout):
        return self.y * (1 - self.y) * dout

class Relu():
    def __init__(self):
        self.mask = None
    def forward(self, x):
        self.mask = (x<=0)
        y = x.copy()
        y[self.mask] = 0
        return y
    def backward(self, dout):
        dout[self.mask] = 0
        return dout

class SoftmaxWithLoss():
    def __init__(self):
        self.y = None
        self.t = None
    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        return cross_entropy_error(self.y, self.t)
    def backward(self, dout):
        batch_size = self.t.shape[0]
        return (self.y - self.t) / batch_size
