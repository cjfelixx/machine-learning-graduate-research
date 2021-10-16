import numpy as np
import matplotlib.pyplot as plt

def KRLS_RFF(u,d,kernel,threshold,D):

    err = np.array([])
    W = np.random.normal(loc=0, scale=2.25, size=(2,D))
    b = np.random.uniform(0,2*np.pi,D).reshape(D,1)
    
    # Initalization
    m = 1
    u_0 = u[0].reshape(2,1)

    h = np.sqrt(2/D) * np.cos(W.T @ u_0 + b)
    k = kernel(u_0,u_0)
    K_inv = np.matrix(1/k)

    P = np.matrix(1)
#     alpha = np.array(d[0]/k).reshape(1,1)
    alpha = np.zeros((D,1))
    
    err = np.append(err,d[0] - h.T @ alpha)
    for n in range(1, len(d)):
        u_n = u[n].reshape(2,1)
        d_n = d[n]
        k = kernel(u_n,u_n)
        h = np.sqrt(2/D) * np.cos(W.T @ u_n + b)
        a = K_inv @ h
        delta = (k - h.T @ a).item()
        err = np.append(err,d_n - h.T @ alpha)
        if delta > threshold:

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

            q_t = (P @ a)/(1 + a.T @ P @ a)
            P = P - ((P @ a @ a.T @ P)/(1 + a.T @ P @ a))

            alpha = alpha + K_inv @ q_t * err[-1]

    return err
