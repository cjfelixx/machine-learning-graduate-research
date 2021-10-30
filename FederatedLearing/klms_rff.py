import numpy as np

def KLMS_RFF(u,d,h,step_size,D,alpha_0):
    
    err = []
    # Initialization
    alpha = alpha_0

    for n in range(len(d)):
        h_n = h.T[n].reshape((D,1))
        err.append((d[n] - h_n.T @ alpha).item()) 
        alpha += step_size * err[-1] * h_n
    return err,alpha