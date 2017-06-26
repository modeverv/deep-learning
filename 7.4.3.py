import sys, os

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + \
                '/deep-learning-from-scratch-master/')
from common.util import im2col,col2im
import numpy as np

x1 = np.random.rand(1,3,7,7)
col1 = im2col(x1,5,5,stride=1,pad=0)
print(col1.shape)

x2 = np.random.rand(10,3,7,7)
col2 = im2col(x2,5,5,stride=1,pad=0)
print(col2.shape)


class Convolution:
    def __init__(self,W,b,stride=1,pad=0):
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad

    def forward(self,x):
        FN,C,FH,FW = self.W.shape
        N,C,H,W = x.shape
        out_h = int(1 + (H + 2 * self.pad - FH) / self.stride)
        out_w = int(1 + (W + 2 * self.pad - FW) / self.stride)

        col = im2col(x,FH,FW,self.stride,self.pad)
        col_w = self.W.reshape(FN, -1).T
        out = np.dot(col,col_w) + self.b
        out = out.reshape(N,out_h,out_w,-1).transpose(0,3,1,2)
        return out

    def backward(self, dout):
        FN, C, FH, FW = self.W.shape
        dout = dout.transpose(0,2,3,1).reshape(-1, FN)

        self.db = np.sum(dout, axis=0)
        self.dW = np.dot(self.col.T, dout)
        self.dW = self.dW.transpose(1, 0).reshape(FN, C, FH, FW)

        dcol = np.dot(dout, self.col_W.T)
        dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)

        return dx

class Pooling:
    def __init__(self,pool_h,pool_w,stride=1,pad=0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad

    def forward(self,x):
        N,C,H,W = x.shape
        out_h = int(1 + (H - self.pool_h) / self.stride)
        out_w = int(1 + (W - self.pool_w) / self.stride)
        # 展開
        col = im2col(x,self.pool_h,self.pool_w,self.stride,self.pad)
        col = col.reshape(-1,self.pool_h * self.pool_w)
        # MAX
        out = np.max(col,axis=1)
        # 整形
        out = out.reshape(N,out_h,out_w,C).transpose(0,3,1,2)

        return out
    