import numpy as np

def softmax(x):
    c = np.max(x, axis=1, keepdims=True)
    fenz = np.exp(x-c)
    fenm = np.sum(fenz, axis=1, keepdims=True)
    return fenz / fenm

def cross_entropy_error(y, t):
    if y.ndim == 1:
        y = y.reshape(1, y.shape)
        t = t.reshape(1, t.shape)
    batch_size = y.shape[0]
    return -np.sum(t * np.log(y+1e-10)) / batch_size