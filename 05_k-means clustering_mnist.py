# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 15:53:49 2018

@author: T901
"""

# %reset
import numpy as np 
from mnist import MNIST # require `pip install python-mnist`
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

from display_network import *

mndata = MNIST('./data/mnist/')
mndata.load_testing()
X = mndata.test_images
print('Number of images: ', len(X))
""" Scaling """
X0 = np.asarray(X)[:1000,:]/256.0
X = X0

K = 10 
kmeans = KMeans(n_clusters=K).fit(X)

pred_label = kmeans.predict(X)

print('Type of cluster center array: ', type(kmeans.cluster_centers_.T))
print('Shape of cluster center array: ', kmeans.cluster_centers_.T.shape)
A = display_network(kmeans.cluster_centers_.T, K, 1)

f1 = plt.imshow(A, interpolation='nearest', cmap = "jet")
f1.axes.get_xaxis().set_visible(False)
f1.axes.get_yaxis().set_visible(False)
plt.show()
# plt.savefig('a1.png', bbox_inches='tight')

# a colormap and a normalization instance
cmap = plt.cm.jet
norm = plt.Normalize(vmin=A.min(), vmax=A.max())

# map the normalized data to colors
# image is now RGBA (512x512x4) 
image = cmap(norm(A))

import scipy.misc
scipy.misc.imsave('aa.png', image)

print('Type of predict label array: ', type(pred_label))
print('Shape of predict label array: ', pred_label.shape)
print('Type of image pixel matrix: ', type(X0))

from sklearn import neighbors

N0 = 20;
X1 = np.zeros((N0*K, 784))
X2 = np.zeros((N0*K, 784))

for k in range(K):
    Xk = X0[pred_label == k, :]

    center_k = [kmeans.cluster_centers_[k]]
    neigh = neighbors.NearestNeighbors(N0).fit(Xk)
    dist, nearest_id  = neigh.kneighbors(center_k, N0)
    
    X1[N0*k: N0*k + N0,:] = Xk[nearest_id, :]     # 20 nearest neighbors of each cluster center
    X2[N0*k: N0*k + N0,:] = Xk[:N0, :]            # 20 first/random neighbors of each cluster center
    
plt.axis('off')
#A = display_network(X2.T, K, N0)
A = display_network(X1.T, K, N0)
f2 = plt.imshow(A, interpolation='nearest' )
plt.gray()
plt.show()

# import scipy.misc
# scipy.misc.imsave('bb.png', A)


# plt.axis('off')
# A = display_network(X1.T, 10, N0)
# scipy.misc.imsave('cc.png', A)
# f2 = plt.imshow(A, interpolation='nearest' )
# plt.gray()

# plt.show()

