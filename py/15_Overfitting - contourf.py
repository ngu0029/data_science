# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 18:17:22 2018

@author: dungnq9
"""

# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals
import math
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(4)

means = [[-1, -1], [1, -1], [0, 1]]
cov = [[1, 0], [0, 1]]
N = 20
X0 = np.random.multivariate_normal(means[0], cov, N)
X1 = np.random.multivariate_normal(means[1], cov, N)
X2 = np.random.multivariate_normal(means[2], cov, N)

X = np.concatenate((X0, X1, X2), axis = 0)
K = 3

original_label = np.asarray([0]*N + [1]*N + [2]*N).T

def kmeans_display(X, label):
    K = np.amax(label) + 1
    X0 = X[label == 0, :]
    X1 = X[label == 1, :]
    X2 = X[label == 2, :]
        
    plt.plot(X0[:, 0], X0[:, 1], 'b^', markersize = 6, alpha = .8)
    plt.plot(X1[:, 0], X1[:, 1], 'go', markersize = 6, alpha = .8)
    plt.plot(X2[:, 0], X2[:, 1], 'rs', markersize = 6, alpha = .8)
    
#   plt.axis('equal')    
    plt.plot()
#   plt.show()

kmeans_display(X, original_label)    
plt.show()
y = original_label.T
X = X.T

def softmax(V):
    e_V = np.exp(V - np.max(V, axis = 0, keepdims = True))
    Z = e_V / e_V.sum(axis = 0)
    return Z

## One-not coding
from scipy import sparse
def convert_labels(y, C = 3):
    Y = sparse.coo_matrix((np.ones_like(y),
                           (y, np.arange(len(y)))), shape = (C, len(y))).toarray()
    return Y

# cost or loss function
lam = 0.001 # regularization parameter
def cost(Y, Yhat, W1, W2, lam):
    return -np.sum(Y*np.log(Yhat))/Y.shape[1] + \
        lam*(np.linalg.norm(W1)**2 + np.linalg.norm(W2)**2)
        
d0 = 2
d1 = h = 100 # size of hidden layer
d2 = C = 3

def mynet(lam):
    # initialize parameters randomly
    W1 = 0.01*np.random.randn(d0, d1)
    b1 = np.zeros((d1, 1))
    W2 = 0.01*np.random.randn(d1, d2)
    b2 = np.zeros((d2, 1))
    
    # X = X.T # each column of X is a data point
    Y = convert_labels(y, C)
    N = X.shape[1]
    eta = 1 # learning rate
    for i in range(10000):
        ## Feedforward 
        Z1 = np.dot(W1.T, X) + b1 
        A1 = np.maximum(Z1, 0)
        Z2 = np.dot(W2.T, A1) + b2
        # import pdb; pdb.set_trace()  # breakpoint 035ab9b5 //
        Yhat = softmax(Z2)

        # compute the loss: average cross-entropy loss
        

        # print loss after each 1000 iterations
        if i %1000 == 0: 
            loss = cost(Y, Yhat, W1, W2, lam)
            print("iter %d, loss: %f" %(i, loss))

        # backpropagation
        E2 = (Yhat - Y )/N
        dW2 = np.dot(A1, E2.T) + lam*W2
        db2 = np.sum(E2, axis = 1, keepdims = True)
        E1 = np.dot(W2, E2)
        E1[Z1 <= 0] = 0 # gradient of ReLU 
        dW1 = np.dot(X, E1.T) + lam*W1
        db1 = np.sum(E1, axis = 1, keepdims = True)

        # Gradient Descent update 
        # import pdb; pdb.set_trace()  # breakpoint 47741f63 //
        W1 += -eta*dW1 
        b1 += -eta*db1 
        W2 += -eta*dW2
        b2 += -eta*db2 
#     return (W1, W2, b1, b2)

    Z1 = np.dot(W1.T, X) + b1
    A1 = np.maximum(Z1, 0)
    Z2 =np.dot(W2.T, A1) + b2
    """
    DUNGNQ9: Returns the indices of the maximum values along an axis.
    This is equivalent to the maximum values using softmax
    """
    predicted_class = np.argmax(Z2, axis=0)
    acc = (100*np.mean(predicted_class == y))
    print('training accuracy: %.2f %%' % acc)

    xm = np.arange(-3, 4, 0.025)
    xlen = len(xm)
    print(xlen)
    ym = np.arange(-4, 4, 0.025)
    ylen = len(ym)
    print(ylen)
    #https://docs.scipy.org/doc/numpy/reference/generated/numpy.meshgrid.html
    xx, yy = np.meshgrid(xm, ym)
    print(xx.shape, yy.shape)

    print(np.ones((1, xx.size)).shape)
    print(np.ones((1, yy.size)).shape)
    #https://docs.scipy.org/doc/numpy/reference/generated/numpy.ravel.html
    xx1 = xx.ravel().reshape(1, xx.size)
    yy1 = yy.ravel().reshape(1, yy.size)

    X0 = np.vstack((xx1, yy1))

    """
    DUNGNQ9: X0.shape = [2, 89600], X0 is used as validation data
    """
    Z1 = np.dot(W1.T, X0) + b1 
    A1 = np.maximum(Z1, 0)
    Z2 = np.dot(W2.T, A1) + b2
    # predicted class 
    """
    DUNGNQ9: Returns the indices of the maximum values along an axis.
    This is equivalent to the maximum values using softmax.
    
    Each element in Z is only one of three values {0, 1, 2}, hence we have only
    three colors for all areas plotted by contourf function.
    The coordinates (xx, yy) that are plotted in the same color mean that they 
    have the same value of Z. (See Level set for more information)
    """
    Z = np.argmax(Z2, axis=0)

    Z = Z.reshape(xx.shape)
    CS = plt.contourf(xx, yy, Z, 200, cmap='jet', alpha = .1)

    kmeans_display(X.T, original_label.T)

    cur_axes = plt.gca()
    cur_axes.axes.get_xaxis().set_ticks([])
    cur_axes.axes.get_yaxis().set_ticks([])
    
    plt.title('$\lambda =$' + str(lam), fontsize = 20)
    fn = 'nnet_reg'+ str(lam) + '.png'
    plt.savefig(fn, bbox_inches='tight', dpi = 600)
    
    plt.show()


# mynet(0)
# mynet(0.1)
# mynet(0.01)
mynet(.1)

    
    
    
    
    