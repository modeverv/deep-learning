import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + \
                '/deep-learning-from-scratch-master/')
import numpy as np
from dataset.mnist import load_mnist



(x_train, t_train), (x_test, t_test) = \
    load_mnist(normalize=True, one_hot_label=True)

print(x_train.shape)
print(t_train.shape)


def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    return - np.sum(t * np.log(y)) / batch_size

