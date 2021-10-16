import numpy as np

def KRLS_RFF(u,d,kernel,l,beta,D):

    err = np.array([])
    W = np.random.normal(loc=0, scale=2.25, size=(2,D))
    b = np.random.uniform(0,2*np.pi,D).reshape(D,1)
    
    # Initalization
    P = np.eye(D)/l
    alpha = np.zeros((D,1))
    for n in range(ken(d)):
        u_n = u[n].reshape(2,1)
        d_n = d[n]
        h = np.sqrt(2/D) * np.cos(W.T @ u_n + b)
        err = np.append(err,d_n - h.T @ alpha)
        
        k = (P @ h)/(beta + h.T @ P @ h)
#         print(k.shape)
        P = (P - (k @ h.T @ P))/beta
        alpha = alpha + k * err[-1]
    return err