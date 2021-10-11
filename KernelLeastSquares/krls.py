import numpy as np
import matplotlib.pyplot as plt

def KRLS(d,kernel,threshold):
    '''
       Kernel Recursive Least Sqaures depends on a kernel function in which to evaluate the points in a
       higher dimension without needing to create and analyaze in the higher dimensional plane as this would
       create significant computing cost.
       The use of kernel functions are valuable as they transform the data into another plane in which they become.
       easy to evaluate.
       For this demonstration, we will be utilizing the guassian kernel function
    '''
    err = np.array([])

    # Initalization
    u = np.matrix([d[0],d[1]])
    dictionary = np.matrix(u)
    h = np.matrix(kernel(u,dictionary))
    k = np.matrix(kernel(u,u))
    K_inv = np.matrix(1/k)

    P = np.matrix(1)
#     alpha = np.matrix(d[0]/k)
    alpha = np.matrix(0)
    m = 1
    err = np.append(err,d[0] - h.T @ alpha)
    for n in range(1, len(d)):
        u_n = np.matrix([d[n-1],d[n]])
        d_n = np.matrix(d[n])
        k = np.matrix(kernel(u_n,u_n))
        h = np.matrix([kernel(u_n,dictionary[j]) for j in range(len(dictionary))]).T
    #     print('k_inv',K_inv)
    #     print('h',h)
        a = np.matrix(K_inv @ h)
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

            alpha = np.matrix(alpha - ((a * err[-1])/delta))
            alpha = np.r_[alpha,[[err[-1]/delta]]]
            m = m + 1
        else:

            q_t = (P @ a)/(1 + a.T @ P @ a)
            P = P - ((P @ a @ a.T @ P)/(1 + a.T @ P @ a))

            alpha = alpha + K_inv @ q_t * err[-1]

#     print('number of SVs',len(dictionary))
    return err
