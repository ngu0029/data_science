# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 12:08:14 2018

@author: T901
"""

import numpy as np
from scipy import misc
np.random.seed(1)

# filename structure
path = './data/YALE/unpadded/'
ids = range(1,16)  # 15 persons
states = ['centerlight', 'glasses', 'happy', 'leftlight',
          'noglasses', 'normal', 'rightlight','sad',
          'sleepy', 'surprised', 'wink' ]
prefix = 'subject'
postfix = '.pgm'

"""
Data dimension is 116 x 98 = 11368 quite big. 
However only the number of images (samples) is 15 x 11 = 165.
So we can compress new data size smaller than 165. This case, choose K = 100.
"""

# data dimension
h = 116 # hight
w = 98 # width
D = h * w
N = len(states)*15
K = 50

# collect all data
X = np.zeros((D, N))
cnt = 0
for person_id in range(1, 16):
    for state in states:
        fn = path + prefix + str(person_id).zfill(2) + '.' + state + postfix
        X[:, cnt] = misc.imread(fn).reshape(D)
        cnt += 1
        
# Doing PCA, note that each row is a datapoint
from sklearn.decomposition import PCA
pca = PCA(n_components=K) # K = 100
pca.fit(X.T)

# projection matrix
""" Attribute components_: Principal axes in feature space, 
representing the directions of maximum variance in the data. 
The components are sorted by explained_variance_.
"""
U = pca.components_.T  
V = pca.explained_variance_.T      

print("\n -first 18 eigen vectors found by PCA")
import matplotlib.pyplot as plt

fig=plt.figure(figsize=(16, 14))
columns = 6
rows = 3
for i in range(1, 19):
    img = U[:, i].reshape((h, w))
    fig.add_subplot(rows, columns, i)
    plt.axis('off')
    img = plt.imshow(img, cmap = 'gray', interpolation='nearest')
    img.axes.get_xaxis().set_visible(False)
    img.axes.get_yaxis().set_visible(False)
plt.show()

import time

start = time.time()

print("\n -Reconstruction of first 6 people")
fig1=plt.figure(figsize=(16, 14))
columns = 6
rows = 2

for person_id in range(6, 11):
    for state in ['centerlight']:
        fn = path + prefix + str(person_id).zfill(2) + '.' + state + postfix
        im = misc.imread(fn)
        subplot_id = person_id - 5
        fig1.add_subplot(rows, columns, subplot_id)
        plt.axis('off')
#         plt.imshow(im, interpolation='nearest' )
        f1 = plt.imshow(im, interpolation='nearest')
        f1.axes.get_xaxis().set_visible(False)
        f1.axes.get_yaxis().set_visible(False)
        plt.gray()
        #fn = 'ori' + str(person_id).zfill(2) + '.png'
        #plt.savefig(fn, bbox_inches='tight', pad_inches=0)
        #plt.show()
        
        # reshape and subtract mean, don't forget 
        """ Attribute mean_: Per-feature empirical mean, estimated from the training set.
            Equal to X.mean(axis=1)
        """
        x = im.reshape(D, 1) - pca.mean_.reshape(D, 1)
        # encode
        """ Projection the image onto lower dimensional space 
            z is a compressed version of x in the lower dimensional space.
        """
        z = U.T.dot(x)
        #decode
        x_tilde = U.dot(z) + pca.mean_.reshape(D, 1)

        # reshape to orginal dim
        im_tilde = x_tilde.reshape(116, 98)
        fig1.add_subplot(rows, columns, subplot_id + 6)
        plt.axis('off')
#         plt.imshow(im_tilde, interpolation='nearest' )
        f1 = plt.imshow(im_tilde, interpolation='nearest')
        f1.axes.get_xaxis().set_visible(False)
        f1.axes.get_yaxis().set_visible(False)
        plt.gray()
        #fn = 'res' + str(person_id).zfill(2) + '.png'
        #plt.savefig(fn, bbox_inches='tight', pad_inches=0)
        #plt.show()              
plt.show()
print("Real image shape = ", x.shape, " compared to compressed image shape =", z.shape)

end = time.time()
print(f'\n Time to complete: {end - start:.2f}s')