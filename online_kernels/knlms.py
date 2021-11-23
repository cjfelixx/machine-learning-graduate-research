import numpy as np

def KNLMS(u,d,kernel,step_size,reg_coeff,threshold,alpha_0=np.array(0).reshape(1,1)):



    err = []
    m = 1

    dictionary = u[0].reshape(1,2)
    h = kernel.fun(u[0],dictionary).reshape(1,1)
    alpha = alpha_0
    
    err.append((d[0] - h.T @ alpha).item())
    for n in range(1, len(d)):
        u_n = u[n].reshape(1,2)

        if np.max(np.abs([kernel.fun(u_n,dictionary[j]) for j in range(len(dictionary))])) <= threshold:
            m += 1
            dictionary = np.r_[dictionary, u_n]
            alpha = np.r_[alpha,[[0]]]

        h = np.array([kernel.fun(u_n,dictionary[j]) for j in range(len(dictionary))]).T.reshape(m,1)
        err.append((d[n] - h.T @ alpha).item())
        alpha = alpha + (step_size/(reg_coeff + (h.T @ h)))*(err[-1] * h)

#     print('number of SVs',len(dictionary))
    return err,h,alpha