# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 11:48:19 2018

@author: T901
"""

import numpy as np
from sklearn import linear_model               # for logistic regression
from sklearn.metrics import accuracy_score     # for evaluation
from scipy import misc                         # for loading image
import scipy.io
np.random.seed(1)                              # for fixing random values

"""
Ref:
https://stackoverflow.com/questions/874461/read-mat-files-in-python    
https://stackoverflow.com/questions/38197449/matlab-data-file-to-pandas-dataframe?noredirect=1&lq=1
"""
mat = scipy.io.loadmat('./data/AR/randomfaces4ar/randomfaces4ar.mat')
#mat = scipy.io.loadmat('randomfaces4ar.mat')
print(type(mat))
# 50 men, 50 women, each taken 26 images, so totally 2600 images
fn = mat['filenameMat']
print(fn.shape)
print(fn[0,0].shape)
print(type(fn), type(fn[0,0]), type(fn[0,0][0,0])) # all np.ndarray - multi-dimensional arrays

# 2600 file names extracted
fn_list = []
mw_no = fn.shape[1]           # total number of men and women
for i in range(mw_no):
    img_no = fn[0,i].shape[1]  # number of images taken for each person
    for j in range(img_no):
        fn_list.append(str(fn[0,i][0,j]).split('[\'')[1].split('\']')[0])
print('Number of images = ', len(fn_list))

# Each photo has 540 features
features = mat['featureMat']
features = np.array(features)
print('Number of features = ', features.shape[0])

# Totally 100 people, each photo is labelled with corrected person among 100 people
# Do not binary-label who is man, who is woman.
labels = mat['labelMat']
labels = np.array(labels)
print('Numbers of labels/persons = ', labels.shape[0])

# generate random projection matrix
D = 165*120     # original dimension
d = features.shape[0]         # new dimension

"""FOLLOWING CODE IGNORED DUE TO ALREADY EXTRACTED DATA"""
"""
# generate the projection matrix 
ProjectionMatrix = np.random.randn(D, d)

# build the file name list
def build_list_fn(pre, img_ids, view_ids):
    """"""
    INPUT:
        pre = 'M-' or 'W-'
        img_ids: indexes of images
        view_ids: indexes of views
    OUTPUT:
        a list of filenames 
    """"""
    list_fn = []
    for im_id in img_ids:
        for v_id in view_ids:
            fn = path + pre + str(im_id).zfill(3) + '-' + \
                str(v_id).zfill(2) + '.bmp'
            list_fn.append(fn)
    return list_fn 
    
def rgb2gray(rgb):
#     Y' = 0.299 R + 0.587 G + 0.114 B 
    return rgb[:,:,0]*.299 + rgb[:, :, 1]*.587 + rgb[:, :, 2]*.114

# feature extraction 
def vectorize_img(filename):    
    # load image 
    rgb = misc.imread(filename)
    # convert to gray scale 
    gray = rgb2gray(rgb)
    # vectorization each row is a data point 
    im_vec = gray.reshape(1, D)
    return im_vec 

def build_data_matrix(img_ids, view_ids):
    total_imgs = img_ids.shape[0]*view_ids.shape[0]*2 
        
    X_full = np.zeros((total_imgs, D))
    y = np.hstack((np.zeros((total_imgs/2, )), np.ones((total_imgs/2, ))))
    
    list_fn_m = build_list_fn('M-', img_ids, view_ids)
    list_fn_w = build_list_fn('W-', img_ids, view_ids)
    list_fn = list_fn_m + list_fn_w 
    
    for i in range(len(list_fn)):
        X_full[i, :] = vectorize_img(list_fn[i])

    X = np.dot(X_full, ProjectionMatrix)
    return (X, y)    
"""
# Rewrite the function build_list_fn()
# Choose the first half men and women for training and 
# the remaining half men and women for testing and
# only images with views that are not covered by glasses and scarfs
train_ids = np.concatenate((np.arange(0, 25), np.arange(50, 75)))
test_ids = np.concatenate((np.arange(25, 50), np.arange(75, 100)))
view_ids = np.hstack((np.arange(0, 7), np.arange(13, 20)))
# Note: np.arange has class of numpy.ndarray, so use concatenate or hstack to combine 2 arrays

def build_list_id(img_ids, view_ids):
    list_id = []
    for im_id in img_ids:
        for v_id in view_ids:
            list_id.append(26*im_id + v_id)
    return list_id 

# Rewrite the function
def build_data_matrix(img_ids, view_ids):
    total_imgs = img_ids.shape[0]*view_ids.shape[0]
    print(total_imgs)
        
    X = np.zeros((d, total_imgs))
    y = np.hstack((np.zeros((int(total_imgs/2), )), np.ones((int(total_imgs/2), ))))
    
    list_id = build_list_id(img_ids, view_ids)
    
    for i in range(total_imgs):
        #print(list_id[i])
        X[:, i] = features[:, list_id[i]]

    return (X.T, y)

(X_train_full, y_train) = build_data_matrix(train_ids, view_ids)
print(X_train_full.shape, y_train.shape)
x_mean = X_train_full.mean(axis = 0)
x_var  = X_train_full.var(axis = 0)
x_std = X_train_full.std(axis = 0)

"""
Standardization -
https://machinelearningcoban.com/general/2017/02/06/featureengineering/#standardization
"""
def feature_extraction(X):
    #return (X - x_mean)/x_var
    return (X - x_mean)/x_std      # standardization

X_train = feature_extraction(X_train_full)
X_train_full = None ## free this variable 

(X_test_full, y_test) = build_data_matrix(test_ids, view_ids)
X_test = feature_extraction(X_test_full)
X_test_full = None

"""
Parameter C: Inverse of regularization strength; must be a positive float. 
Like in support vector machines, smaller values specify stronger regularization.
See: https://machinelearningcoban.com/2017/03/04/overfitting/#-regularization
"""
logreg = linear_model.LogisticRegression(C=1e5) # just a big number 
logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)
print("Accuracy: %.2f %%" %(100*accuracy_score(y_test, y_pred)))

""" Test with some images """
#fn1 = 'M-036-18.bmp'
#fn2 = 'W-045-01.bmp'
#fn3 = 'M-048-01.bmp'
#fn4 = 'W-027-02.bmp'

# First 0..49 indices are men
fn1_id = 26*(36-1) + (18-1)
fn2_id = 26*(50+45-1) + (1-1)
fn3_id = 26*(48-1) + (1-1)
fn4_id = 26*(50+27-1) + (2-1)

x1 = feature_extraction(features[:, fn1_id].T)
p1 = logreg.predict_proba(x1)
print(p1)

x2 = feature_extraction(features[:, fn2_id].T)
p2 = logreg.predict_proba(x2)
print(p2)

x3 = feature_extraction(features[:, fn3_id].T)
p3 = logreg.predict_proba(x3)
print(p3)

x4 = feature_extraction(features[:, fn4_id].T)
p4 = logreg.predict_proba(x4)
print(p4)


