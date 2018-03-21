# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 15:29:49 2017

@author: dungnq9
"""

# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals
import numpy as np 
import matplotlib.pyplot as plt

"""
mapping:
    Times City = 0
    Royal City = 1
    KDT Linh Đàm = 2
    
Enter values of housing input parameters into csv file
Convert all our categorical variables into numeric by encoding the categories 
using LabelEncoder module of sklearn.preprocessing
"""



# Housing input parameters
X = np.array([[3, 2000, 0], 
              [2, 800, 1],
              [2, 850, 0],
              [1, 550, 0],
              [4, 2000, 2]])
# Price ($)
y = np.array([[ 250000, 300000, 150000,  78000, 150000]]).T

# Building Xbar 
one = np.ones((X.shape[0], 1))
Xbar = np.concatenate((one, X), axis = 1)

""" Using scikit-learn library """
from sklearn import datasets, linear_model

# fit the model by Linear Regression
regr = linear_model.LinearRegression(fit_intercept=False) # fit_intercept = False for calculating the bias
regr.fit(Xbar, y)

# Compare two results
print( 'Solution found by scikit-learn  : ', regr.coef_ )

xx0 = 3
xx1 = 2000
xx2 = 1 #Royal City

print(regr.coef_.shape)

yy = regr.coef_[0][0] + regr.coef_[0][1]*xx0 + regr.coef_[0][2]*xx1 + regr.coef_[0][3]*xx2
print(yy)