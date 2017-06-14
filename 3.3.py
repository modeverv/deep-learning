# coding: utf-8

import numpy as np

A = np.array([1, 2, 3, 4])
print(A)
print(np.ndim(A))
print(A.shape)
print(A.shape[0])

B = np.array([[1, 2], [3, 4], [5, 6]])

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

C = np.dot(A, B)

X = np.array([[1,2]])
W = np.array([[1,3,5],[2,4,6]])
print(X.shape)
print(W.shape)
Y = np.dot(X,W)
print(Y)
