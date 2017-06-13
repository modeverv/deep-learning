# coding: utf-8

import numpy as np
import matplotlib.pylab as plt


def step_function(x):
	return np.array(x > 0, dtype=np.int)


x = np.arange(-5.0, 5.0, 0.1)
y = step_function(x)


# plt.plot(x,y)
# plt.ylim(-0.1,1.1)
# plt.show()

def sigmoid(x):
	return 1 / (1 + np.exp(-x))


y = sigmoid(x)


# plt.plot(x,y)
# plt.ylim(-0.1,1.1)
# plt.show()

def xxx(x):
	return x * x * x + 2 * x + 1


y = xxx(x)


# plt.plot(x,y)
# plt.ylim(-500.1,500.1)
# plt.show()

def relu(x):
	return np.maximum(0, x)
