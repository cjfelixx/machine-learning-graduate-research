import numpy as np
import matplotlib.pyplot as plt

def KNLMS(d,kernel,step_size,reg_coeff,threshold):
    
    err = np.array([])
    
    # Initialization
    m = 1
    u = np.array([d[0],d[1]])
    dictionary = np.array(u).reshape(1,2)
    h = np.array(kernel(u,dictionary)).reshape(1,1)
    alpha = np.array(0).reshape(1,1)
    
    err = np.append(err,d[0] - h.T @ alpha)
    for n in range(1, len(d)):
        u_n = np.array([d[n-1],d[n]]).reshape(1,2)
        d_n = d[n]

        if np.max(np.abs([kernel(u_n,dictionary[j]) for j in range(len(dictionary))])) < threshold:
            m += 1
            dictionary = np.r_[dictionary, u_n]
            h = np.matrix([kernel(u_n,dictionary[j]) for j in range(len(dictionary))]).T.reshape(m,1)
            alpha = np.r_[alpha,[[0]]]

        h = np.array([kernel(u_n,dictionary[j]) for j in range(len(dictionary))]).T.reshape(m,1)
        alpha = alpha + (step_size/(reg_coeff + (np.linalg.norm(h,ord=2)**2)))*((d_n - h.T @ alpha).item() * h)
        err = np.append(err,d_n - h.T @ alpha)

#     print('number of SVs',len(dictionary))
    return err