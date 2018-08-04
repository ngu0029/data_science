# -*- coding: utf-8 -*-
"""
Created on Sat Apr 28 11:59:08 2018

@author: T901
"""

# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals
# list of points
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
np.random.seed(22)

means = [[0, 5], [5, 0]]
cov0 = [[4, 3], [3, 4]]
cov1 = [[3, 1], [1, 1]]
N0 = 50
N1 = 40
N = N0 + N1
X0 = np.random.multivariate_normal(means[0], cov0, N0) # each row is a data point
X1 = np.random.multivariate_normal(means[1], cov1, N1)

# Build S_B = (m1-m2)(m1-m2)^T
m0 = np.mean(X0.T, axis = 1, keepdims = True)
m1 = np.mean(X1.T, axis = 1, keepdims = True)

a = (m0 - m1)
S_B = a.dot(a.T)

# Build S_W = (xn_1-m1)(xn_1-m1)^T + (xn_2-m2)(xn_2-m2)^T
A = X0.T - np.tile(m0, (1, N0))
B = X1.T - np.tile(m1, (1, N1))

S_W = A.dot(A.T) + B.dot(B.T)

# Find solution of W
e_val, W = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))
w = W[:,0]
print(e_val)
print(w)

# Compare with solution found by sklearn
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
X = np.concatenate((X0, X1))
y = np.array([0]*N0 + [1]*N1)
clf = LinearDiscriminantAnalysis()
clf.fit(X, y)

print("\n",clf.coef_)
print(clf.coef_/np.linalg.norm(clf.coef_)) # normalize