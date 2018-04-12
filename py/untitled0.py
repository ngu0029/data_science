# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 17:49:42 2018

@author: dungnq9
"""
"""
import sys

a = range(1,6)
print(sys.getsizeof(a))
for i in range (1,6):
    print(i)
#for i in xrange (1,6):
    #print(i)
"""

"""    
import numpy as np    
import matplotlib.pyplot as plt

mean = [0, 0]
cov = [[1, 0], [0, 100]]  # diagonal covariance

x, y = np.random.multivariate_normal(mean, cov, 5000).T
plt.plot(x, y, 'x')
plt.axis('equal')
plt.show()    
"""

"""
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

x = np.arange(-5, 5, 0.1)
y = np.arange(-5, 5, 0.1)
xx, yy = np.meshgrid(x, y, sparse=True)
z = np.sin(xx**2 + yy**2) / (xx**2 + yy**2)
h = plt.contourf(x,y,z)
#g = plt.contour(x,y,z)