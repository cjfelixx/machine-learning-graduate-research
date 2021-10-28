import numpy as np

def KRLS_RFF(u,d,kernel_params,D,alpha_0,beta=1.0,l=1.0):

    sigma = 1/np.sqrt(2*kernel_params.sigma)
    
    err = np.array([])
    W = (1/sigma) * np.random.normal(loc=0, scale=sigma, size=(2,D))
    b = np.random.uniform(0,2*np.pi,D).reshape(D,1)
    
    # Initalization
    P = np.eye(D)/l
    alpha = alpha_0
    for n in range(len(d)):
        u_n = u[n].reshape(2,1)
        d_n = d[n]
        h = np.sqrt(2/D) * np.cos(W.T @ u_n + b)
        err = np.append(err,d_n - h.T @ alpha)
        
        k = (P @ h)/(beta + h.T @ P @ h)
#         print(k.shape)
        P = (P - (k @ h.T @ P))/beta
        alpha = alpha + k * err[-1]
    return err,h,alpha