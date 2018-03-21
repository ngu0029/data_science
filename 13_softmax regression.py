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
