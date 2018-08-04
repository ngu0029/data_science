# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 12:09:24 2018

@author: T901
"""

import numpy as np
from numpy import linalg as LA
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

def do_svd(A):
    U, S, V = LA.svd(A)
    
    m = A.shape[0]
    n = A.shape[1]
    
    # checking if U, V are orthogonal and S is a diagonal matrix with
    # nonnegative descreasing elements
    print('Probenius norm of (UU^T - I) = ', LA.norm(U.dot(U.T) - np.eye(m)))
    
    print('\n S = ', S, '\n')
    
    print('Probenius norm of (VV^T - I) = ', LA.norm(V.dot(V.T) - np.eye(n)))
    
    # S contains only diagonal values, V is indeed V.T
    return U, S, V

print('\n-For simple matrix 2 x 3:')
m, n = 2, 3
A = np.random.rand(m, n)
U, S, V = do_svd(A)

print('\n-For a stored image:')
img = mpimg.imread('./data/tdt_building.jpg')
plt.imshow(img)
plt.axis('off')
plt.show()


# to gray 
i = 0.2125* img[:, :, 0] + 0.7154 *img[:, :, 1] + 0.0721 *img[:, :, 2]
plt.imshow(i, cmap = 'gray')
plt.axis('off')
plt.show()

print(type(i), i.shape)
U, S, V = do_svd(i)

from matplotlib.backends.backend_pdf import PdfPages

with PdfPages('singular_value.pdf') as pdf:
    plt.semilogy(S) 
    plt.xlabel('$k$', fontsize = 20)
    plt.ylabel('$\sigma_k$', fontsize = 20)
    # We change the fontsize of minor ticks label 
    plt.tick_params(axis='both', labelsize=18)
    plt.tight_layout()
    pdf.savefig()
    plt.show()
    
# percentage of preserving energy

with PdfPages('energy_preserved.pdf') as pdf:
    a = np.sum(S**2)
    b = np.zeros_like(S)
    for i in range(S.shape[0]):
        b[i] = np.sum(S[:i+1]**2, axis = 0)/a

    plt.plot(b)
    plt.xlabel('$k$', fontsize = 20)
#    plt.ylabel('$\|\|\mathbf{A} - \mathbf{A}_k\|\|_F^2 / \|\|\mathbf{A}\|\|_F^2$', fontsize = 20)
    plt.ylabel('$\|\|\mathbf{A}_k\|\|_F^2 / \|\|\mathbf{A}\|\|_F^2$', fontsize = 20)
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.tight_layout()
    pdf.savefig()
    plt.show()

# Show compressed image with different values of k
def approx_rank_k(U, S, V, k):
    Uk = U[:, :k]
    Sk = S[:k]
    Vk = V[:k, :]
    return np.around(Uk.dot(np.diag(Sk)).dot(Vk))

## error
e =  1 - b

# A = gray
# U, S, V = LA.svd(A)
A1 = []
for k in range(5, 100, 10):
    A1.append(approx_rank_k(U, S, V, k))

# show results
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation 
fig, ax = plt.subplots()
def update(i):
    ani = plt.cla()
    ani = plt.imshow(A1[i], cmap = 'gray') # display in grey scale
    label = '$k$ = %d: error = %.4f' %(10*i + 5, e[i])
    ax.set_xlabel(label)
    ani = plt.axis('off')
    ani = plt.title(label)

    return ani, ax 

anim = FuncAnimation(fig, update, frames=np.arange(0, len(A1)), interval=500)
#anim.save('a.mp4', dpi = 300, writer = 'ffmpeg')
"""
https://stackoverflow.com/questions/46712669/exporting-animated-gif-using-matplotlib?rq=1
http://tiao.io/posts/notebooks/save-matplotlib-animations-as-gifs/
"""
plt.rcParams["animation.convert_path"] = "C:\Program Files\ImageMagick-7.0.7-Q16\magick.exe"
anim.save('tdt_building.gif', dpi = 300, writer = 'imagemagick', extra_args="convert_path")
plt.show()
