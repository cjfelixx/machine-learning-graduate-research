import numpy as np
import matplotlib.pyplot as plt

def KRLS(u,d,kernel_params,threshold,alpha_0=np.matrix(0).reshape(1,1),beta=1.0):
    '''
       Kernel Recursive Least Sqaures depends on a kernel function in which to evaluate the points in a
       higher dimension without needing to create and analyaze in the higher dimensional plane as this would
       create significant computing cost.
       The use of kernel functions are valuable as they transform the data into another plane in which they become.
       easy to evaluate.
       For this demonstration, we will be utilizing the guassian kernel function
    '''
    sigma = kernel_params.sigma
    kernel = lambda u_i,u_j: np.exp(-1 * sigma * (np.linalg.norm(u_i - u_j,ord=2)**2))    
    err = np.array([])    

    # Initalization
    m = 1
    u_0 = u[0]
    dictionary = np.array(u_0).reshape(1,2)
    h = np.array(kernel(u_0,dictionary)).reshape(1,1)
    k = kernel(u_0,u_0)
    K_inv = np.matrix(1/k)

    P = np.matrix(1)
#     alpha = np.array(d[0]/k).reshape(1,1)
    alpha = alpha_0
    err = np.append(err,d[0] - h.T @ alpha)
    for n in range(1, len(d)):
        u_n = u[n].reshape(1,2)
        d_n = d[n]
        k = kernel(u_n,u_n)
        h = np.array([kernel(u_n,dictionary[j]) for j in range(len(dictionary))]).T.reshape(m,1)
        a = K_inv @ h
        delta = (k - h.T @ a).item()
        err = np.append(err,d_n - h.T @ alpha)
        if delta > threshold:
            dictionary = np.r_[dictionary, u_n]

            K_inv_num = np.c_[delta*K_inv + a @ a.T,-a]
            K_inv_den = np.c_[-a.T, 1]
            K_inv = np.r_[K_inv_num,K_inv_den]
            K_inv = K_inv/delta

            P_num = np.c_[P,np.zeros((m,1))]
            P_den = np.c_[np.zeros((m,1)).T, 1]
            P = np.r_[P_num,P_den]

            alpha = np.array(alpha - ((a * err[-1])/delta)).reshape(m,1)
            alpha = np.r_[alpha,[[err[-1]/delta]]]
            m = m + 1
        else:

            q_t = (P @ a)/(beta + a.T @ P @ a)
            P = (P - ((P @ a @ a.T @ P)/(beta + a.T @ P @ a)))/beta
            alpha = alpha + K_inv @ q_t * err[-1]

#     print('number of SVs',len(dictionary))
    return err
