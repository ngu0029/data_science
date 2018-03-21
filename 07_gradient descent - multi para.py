# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 17:13:13 2018
This program is to check gradient of a ML algorithm btw formula and numerical way
@author: dungnq9
"""
import numpy as np
np.random.seed(2)

X = np.random.rand(1000, 1)
y = 4 + 3*X + .2*np.random.randn(1000, 1) # noise added

# Building Xbar
one = np.ones((X.shape[0],1))
Xbar = np.concatenate((one, X), axis = 1)

""" gradient of loss function of Linear Regression using formula """
""" Gradient of a scalar function over a vector is A VECTOR WITH THE SAME SIZE """
def grad(w):
    N = Xbar.shape[0]
    h = 1/N * Xbar.T.dot(Xbar.dot(w) - y)
    #print("h = ", h)
    return h

""" loss function of Linear Regression """
def cost(w):
    N = Xbar.shape[0]
    return .5/N*np.linalg.norm(y - Xbar.dot(w), 2)**2

""" Numerical gradient is applied for each variable while the others are kept fixed  """
def numerical_grad(w, cost):
    eps = 1e-4
    g = np.zeros_like(w)
    for i in range(len(w)):
        w_p = w.copy()
        w_n = w.copy()
        w_p[i] += eps
        w_n[i] -= eps
        g[i] = (cost(w_p) - cost(w_n))/(2*eps)
        print("\nw_p = ", w_p, "\nw_n = ", w_n)
    print("g = ", g)        
    return g

def check_grad(w, cost, grad):
    w_rand2 = np.random.rand(w.shape[0], w.shape[1])
    print("w_rand2 = ", w_rand2)
    grad1 = grad(w_rand2)
    grad2 = numerical_grad(w_rand2, cost)
    return True if np.linalg.norm(grad1 - grad2) < 1e-6 else False

w_rand1 = np.random.rand(2,1)
print("w_rand1 = ", w_rand1)
print('Checking gradient...', check_grad(w_rand1, cost, grad))

# After gradient check is all right, use grad formula to find w parameter
def myGD(w_init, grad, eta):
    w = [w_init]
    for it in range(100):
        w_new = w[-1] -eta*grad(w[-1])
        if np.linalg.norm(grad(w_new))/len(w_new) < 1e-3:
            break
        w.append(w_new)
    return (w, it)

w_init = np.array([[2], [1]])
(w1, it1) = myGD(w_init, grad, 1)
print("Solution found by GD: w = ", w1[-1].T, ",\nafter %d iterations." %(it1+1))
        
    
    