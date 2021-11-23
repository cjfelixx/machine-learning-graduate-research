import numpy as np
import matplotlib.pyplot as plt

def KRLS(u,d,kernel,threshold,alpha_0=np.matrix(0).reshape(1,1),beta=1.0):

    err = []
    m = 1
    dictionary = u[0].reshape(1,2)
    h = kernel.fun(u[0],dictionary).reshape(1,1)
    k = kernel.fun(u[0],u[0])
    K_inv = np.matrix(1/k)

    P = np.matrix(1)
#     alpha = np.array(d[0]/k).reshape(1,1)
    alpha = alpha_0
    err.append((d[0] - h.T @ alpha).item())
    for n in range(1, len(d)):
        u_n = u[n].reshape(1,2)
        k = kernel.fun(u_n,u_n)
        h = np.array([kernel.fun(u_n,dictionary[j]) for j in range(len(dictionary))]).T.reshape(m,1)
        a = K_inv @ h
        delta = (k - h.T @ a).item()
        err.append((d[n] - h.T @ alpha).item())
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
            P_a = P @ a
            a_P_a = a.T @ P_a
            q_t = (P_a)/(beta + a_P_a)
            P = (P - ((P_a @ a.T @ P)/(beta + a_P_a)))/beta
            alpha = alpha + K_inv @ q_t * err[-1]

#     print('number of SVs',len(dictionary))
    return err,h,alpha