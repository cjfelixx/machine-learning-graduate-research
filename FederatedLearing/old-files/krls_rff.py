import numpy as np

def KRLS_RFF(u,d,h,D,alpha_0,beta=1.0,l=1.0):
   
    err = []
    P = np.eye(D)/l
    alpha = alpha_0
    for n in range(len(d)):
        h_n = h.T[n].reshape((D,1))
        err.append((d[n] - h_n.T @ alpha).item())  
        a = h_n.T @ P
        k = (P @ h_n)/(beta + a @ h_n)
        P = (P - (k @ a))/beta
        alpha = alpha + k * err[-1]
    return err,alpha