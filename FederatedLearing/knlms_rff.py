import numpy as np

def KNLMS_RFF(u,d,h,step_size,reg_coeff,D,alpha_0):
    
    err = []
    # Initialization
    alpha = alpha_0
    for n in range(len(d)):
        h_n = h.T[n].reshape((D,1))
        err.append((d[n] - h_n.T @ alpha).item()) 
        alpha = alpha + (step_size/(reg_coeff + (h_n.T @ h_n))) * (err[-1] * h_n)
    return err,alpha