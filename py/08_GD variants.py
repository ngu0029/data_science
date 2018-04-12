# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 11:31:38 2018

@author: dungnq9
"""
import numpy as np

# test
w = np.array([[2], [2]])
print(len(w))
print(np.linalg.norm(w))
print(np.linalg.norm(w)/len(w))
print(np.linalg.norm(w)**2)

print(np.zeros_like(w))

# check convergence
def has_converged(theta_new, grad):
    return np.linalg.norm(grad(theta_new))/len(theta_new) < 1e-3

""" Momentum GD """                          
def GD_momentum(theta_init, grad, eta, gamma):
    # Suppose we want to store history of theta
    theta = [theta_init]
    v_old = np.zeros_like(theta_init)
    for it in range(100):
        v_new = gamma*v_old + eta*grad(theta[-1])
        theta_new = theta[-1] - v_new
        theta.append(theta_new)
        if has_converged(theta_new, grad):
            break
        v_old = v_new
    return theta
    # this variable includes all points in the path
    # if you just want the final answer
    # use 'return theta[-1]'
    
""" Nesterov accelerated gradient (NAG) """
def GD_NAG(theta_init, grad, eta, gamma):
    # SUppose we want to stor history of theta
    theta = [theta_init]
    v_old = np.zeros_like(theta_init)
    for it in range(100):
        v_new = gamma*v_old + eta*grad(theta[-1] - gamma*v_old)
        theta_new = theta[-1] - v_new
        theta.append(theta_new)
        if has_converged(theta_new, grad):
            break
        v_old = v_new
    return theta

""" Batch Gradient Descent """
""" Gradient is calculated on all data points for each loop """

""" Stochastic Gradient Descent """
""" At a time, gradient is only calculated based on only one data point,
    then update w based on this gradient
"""
"""
SGD chỉ yêu cầu một lượng epoch rất nhỏ (thường là 10 cho lần đầu tiên, 
sau đó khi có dữ liệu mới thì chỉ cần chạy dưới một epoch là đã có nghiệm tốt). 
Vì vậy SGD phù hợp với các bài toán có lượng cơ sở dữ liệu lớn 
(chủ yếu là Deep Learning mà chúng ta sẽ thấy trong phần sau của blog) 
và các bài toán yêu cầu mô hình thay đổi liên tục, tức online learning.

Một điểm cần lưu ý đó là: sau mỗi epoch, chúng ta cần shuffle (xáo trộn) 
thứ tự của các dữ liệu để đảm bảo tính ngẫu nhiên. Việc này cũng ảnh hưởng 
tới hiệu năng của SGD.
"""
# single point gradient
def sgrad(w, i, rd_id):
    true_i = rd_id[i]
    xi = Xbar[true_i, :]
    yi = y[true_i]
    a = np.dot(xi, w) - yi
    return (xi*a).reshape(2, 1)

def SGD(w_init, sgrad, eta):
    w = [w_init]
    w_last_check = w_init
    iter_check_w = 10
    N = X.shape[0]
    count = 0
    for it in range(10):
        # shuffle data
        rd_id = np.random.permutation(N)
        for i in range(N):
            count += 1
            g = sgrad(w[-1], i, rd_id)
            w_new = w[-1] - eta*g
            w.append(w_new)
            if count%iter_check_w == 0:
                w_this_check = w_new
                if np.linalg.norm(w_this_check - w_last_check)/len(w_init) < 1e-3:
                    return w
                w_last_check = w_this_check
    return w

""" Mini_batch Gradient Descent """
""" The code below is applied for Linear Regression 
    For other algorithms, we need to update Gradient function
"""

X = np.random.rand(1000, 1)
y = 4 + 3*X + .2*np.random.randn(1000, 1) # noise added

# Building Xbar
one = np.ones((X.shape[0],1))
Xbar = np.concatenate((one, X), axis = 1)

# mini-batch gradient
def mbgrad(w, istart, iend, rd_id):
    range_i = rd_id[istart:iend]
    Xi = Xbar[range_i, :]
    Yi = y[range_i]
    A = np.dot(Xi, w) - Yi
    return np.dot(Xi.T, A).reshape(Xbar.shape[1], 1)

def mbGD(w_init, mbgrad, eta):
    w = [w_init]
    w_last_check = w_init
    iter_check_w = 10
    N = X.shape[0]
    batch_size = 50
    no_of_batch = int(N/batch_size)
    print(no_of_batch)
    count = 0
    for it in range(10):
        # shuffle data
        rd_id = np.random.permutation(N)
        for i in range(no_of_batch):
            count += 1
            g = mbgrad(w[-1], i*50, i*50 + 49, rd_id)
            print("g", i, " = ", g)
            w_new = w[-1] - eta*g
            w.append(w_new)
            if count%iter_check_w == 0:
                w_this_check = w_new
                """
                Stopping criteria: compare w solutions after 10 updates
                """
                if np.linalg.norm(w_this_check - w_last_check)/len(w_init) < 1e-3:
                    return w
                w_last_check = w_this_check
    return w
""" Great learning rate eta causes the algorithm to be diverged, NOT converged """
#w_list = mbGD(np.random.rand(2,1), mbgrad, 1)
#w_list = mbGD(np.random.rand(2,1), mbgrad, 0.1)
w_list = mbGD(np.random.rand(2,1), mbgrad, 0.01)
#print("Result of mini-batch GD = ", w_list)
print("Parameter w of mini-batch GD = ", w_list[-1])