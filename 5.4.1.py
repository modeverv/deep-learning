import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + \
                '/deep-learning-from-scratch-master/')
import numpy as np
from common.functions import softmax,cross_entropy_error


class MulLayer:
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self,x,y):
        self.x = x
        self.y = y
        out = x * y
        return out

    def backward(self,dout):
        dx = dout * self.y
        dy = dout * self.x
        return dx,dy


class AddLayer:
    def __init__(self):
        pass

    def forward(self,x,y):
        out = x + y
        return out

    def backword(self,dout):
        dx = dout * 1
        dy = dout * 1
        return dx,dy

apple = 100
apple_num = 2
orange = 150
orange_num = 3
tax = 1.1

# Layer
mul_apple_layer = MulLayer()
mul_orange_layer = MulLayer()
add_apple_orange_layer = AddLayer()
mul_tax_layer = MulLayer()

# forward
apple_price = mul_apple_layer.forward(apple,apple_num)
orange_price = mul_orange_layer.forward(orange,orange_num)
all_price = add_apple_orange_layer.forward(apple_price,orange_price)
price = mul_tax_layer.forward(all_price,tax)

# backward
dprice = 1
dall_price , dtax = mul_tax_layer.backward(dprice)
dapple_price,dorange_price = add_apple_orange_layer.backword(dall_price)
dorange,dorange_num = mul_orange_layer.backward(dorange_price)
dapple,dapple_num = mul_apple_layer.backward(dapple_price)

print(price)
print(dapple_num,dapple)
print(dorange_num,dorange)
print(dtax)


class Relu:
    def __init__(self):
        self.mask = None

    def forward(self,x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        return out

    def backward(self,dout):
        dout[self.mask] = 0
        dx = dout
        return dx


# http://nonbiri-tereka.hatenablog.com/entry/2014/06/30/134023
class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self,x):
        out = 1 / (1 + np.exp(-x))
        self.out = out
        return out

    def backward(self,dout):
        dx = dout * (1.0 - self.out) * self.out


class Affine:
    def __init__(self,W,b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None

    def forward(self,x):
        self.x = x
        out = np.dot(x,self.W) + self.b
        return out

    def backward(self,dout):
        dx = np.dot(dout,self.W.T)
        self.dW = np.dot(self.x.T,dout)
        self.db = np.sum(dout,axis=0)
        return dx

class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None

    def forward(self,x,t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y,self.t)
        return self.loss

    def backward(self,dout=1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size
        return dx
