# -*- coding: utf-8 -*-
"""
Created on Mon Dec 25 14:52:34 2017

@author: dungnq9
"""

# generate data
# list of points 
import numpy as np 
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
np.random.seed(2)

means = [[2, 2], [4, 2]]
cov = [[.3, .2], [.2, .3]]
N = 10
X0 = np.random.multivariate_normal(means[0], cov, N).T
X1 = np.random.multivariate_normal(means[1], cov, N).T

X = np.concatenate((X0, X1), axis = 1)
y = np.concatenate((np.ones((1, N)), -1*np.ones((1, N))), axis = 1)
# Xbar 
X = np.concatenate((np.ones((1, 2*N)), X), axis = 0)

"""
h(w,x): calculate output knowing input x and weights w
"""
def h(w, x):
    return np.sign(np.dot(w.T, x))

"""
has_converged(X, y, w): check if the algo is converged, by comparing h(w, X) 
with ground true y. If same, stop the run.
"""
def has_converged(X, y, w):
    return np.array_equal(h(w, X), y)

"""
perception(X, y, w_init): main function of PLA
"""
def perceptron(X, y, w_init):
    w = [w_init]
    print("w=",w)
    N = X.shape[1]
    d = X.shape[0]
    mis_points = []
    while True:
        #mix data
        mix_id = np.random.permutation(N)
        for i in range(N):
            xi = X[:, mix_id[i]].reshape(d, 1)
            yi = y[0, mix_id[i]]
            print("xi=",xi,", yi=",yi," w[-1]=",w[-1])
            """w[-1] refers to the last element"""
            if h(w[-1], xi)[0] != yi: #misclassified point
                mis_points.append(mix_id[i])
                w_new = w[-1] + yi*xi
                w.append(w_new)
            
        if has_converged(X, y, w[-1]):
            break
    return (w, mis_points)

d = X.shape[0]
w_init = np.random.randn(d, 1)
(w, m) = perceptron(X, y, w_init)


            
            
    
