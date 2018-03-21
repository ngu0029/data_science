# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 17:45:01 2018

@author: T901
"""

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

img = mpimg.imread('./data/girl3.jpg')
#plt.imshow(img)
imgplot = plt.imshow(img)
plt.axis('off')
plt.show() 

X = img.reshape((img.shape[0]*img.shape[1], img.shape[2]))

for K in [3]:
    kmeans = KMeans(n_clusters=K).fit(X)
    label = kmeans.predict(X)
    print(label.shape)

    img4 = np.zeros_like(X)
    print(img4.shape)
    # replace each pixel by its center
    # label == k will return all indice in vector label that have elements equal to k
    for k in range(K):
        img4[label == k] = kmeans.cluster_centers_[k]
    # reshape and display output image
    img5 = img4.reshape((img.shape[0], img.shape[1], img.shape[2]))
    plt.imshow(img5, interpolation='nearest')
    plt.axis('off')
    plt.show()

"""    
for K in [2, 5, 10, 15, 20]:
    kmeans = KMeans(n_clusters=K).fit(X)
    label = kmeans.predict(X)

    img4 = np.zeros_like(X)
    # replace each pixel by its center
    for k in range(K):
        img4[label == k] = kmeans.cluster_centers_[k]
    # reshape and display output image
    img5 = img4.reshape((img.shape[0], img.shape[1], img.shape[2]))
    plt.imshow(img5, interpolation='nearest')
    plt.axis('off')
    plt.show()    
"""