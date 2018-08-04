# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 16:35:06 2018

@author: dungnq9
"""

import numpy as np 

def softmax(Z):
    """
    Compute softmax values for each sets of scores in V.
    each column of V is a set of score.    
    """
    e_Z = np.exp(Z)
    A = e_Z / e_Z.sum(axis = 0)
    return A

def softmax_stable(Z):
    """
    Compute softmax values for each sets of scores in V.
    each column of V is a set of score.    
    """
    max_Z = np.max(Z, axis = 0, keepdims = True)
    #print(max_Z)
    e_Z = np.exp(Z - max_Z)
    A = e_Z / e_Z.sum(axis = 0)
    return A    

Z = np.array([3, 3, .1])
A = softmax_stable(Z)

def softmax_regression(X, y, W_init, eta, tol = 1e-4, max_count = 10000):
    W = [W_init]    
    C = W_init.shape[1]
    Y = convert_labels(y, C)
    it = 0
    N = X.shape[1]
    d = X.shape[0]
    
    count = 0
    check_w_after = 20
    while count < max_count:
        # mix data 
        mix_id = np.random.permutation(N)
        for i in mix_id:
            xi = X[:, i].reshape(d, 1)
            yi = Y[:, i].reshape(C, 1)
            ai = softmax(np.dot(W[-1].T, xi))
            W_new = W[-1] + eta*xi.dot((yi - ai).T)
            count += 1
            # stopping criteria
            if count%check_w_after == 0:                
                if np.linalg.norm(W_new - W[-check_w_after]) < tol:
                    return W
            W.append(W_new)
    return W

import numpy as np 

## One-hot coding
from scipy import sparse 
def convert_labels(y, C = 3):
    """
    convert 1d label to a matrix label: each column of this 
    matrix coresponding to 1 element in y. In i-th column of Y, 
    only one non-zeros element located in the y[i]-th position, 
    and = 1 ex: y = [0, 2, 1, 0], and 3 classes then return

            [[1, 0, 0, 1],
             [0, 0, 1, 0],
             [0, 1, 0, 0]]
    """
    Y = sparse.coo_matrix((np.ones_like(y), 
        (y, np.arange(len(y)))), shape = (C, len(y))).toarray()
    return Y 

# randomly generate data 
N = 2 # number of training sample 
d = 2 # data dimension 
C = 3 # number of classes 

X = np.random.randn(d, N)  # shape (2, 2)
y = np.random.randint(0, 3, (N,))

X = np.concatenate((np.ones((1, N)), X), axis = 0)  # shape (d+1, N) = (3, N)

Y = convert_labels(y, C)

eta = .05
W_init = np.random.randn(X.shape[0], C)

W = softmax_regression(X, y, W_init, eta)  # shape (d + 1, C) = (3, 3)
# W[-1] is the solution, W is all history of weights
print(W[-1])


"""For number of samples = 3 -- X1, X2, X3"""
#means = [[2, 2], [8, 3], [3, 6]]
#cov = [[1, 0], [0, 1]]
#N = 500
#X0 = np.random.multivariate_normal(means[0], cov, N)  # shape (500, 2), d = 2
#X1 = np.random.multivariate_normal(means[1], cov, N)
#X2 = np.random.multivariate_normal(means[2], cov, N)
#
## each column is a datapoint
#X = np.concatenate((X0, X1, X2), axis = 0).T  # shape (2, 1500)
## extended data
#X = np.concatenate((np.ones((1, 3*N)), X), axis = 0)  # shape (3, 1500)
#C = 3
#
#original_label = np.asarray([0]*N + [1]*N + [2]*N).T
#
#W_init = np.random.randn(X.shape[0], C)
#W = softmax_regression(X, original_label, W_init, eta)  # shape (d + 1, C) = (3, 3)
#print()
#print(W[-1])