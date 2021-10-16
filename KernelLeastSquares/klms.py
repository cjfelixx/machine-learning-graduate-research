import numpy as np

def KLMS(u,d,kernel,step_size,threshold):
    
    err = np.array([])
    
    # Initialization
    m = 1
    u_0 = u[0]
    dictionary = np.array(u_0).reshape(1,2)
    h = np.array(kernel(u_0,dictionary)).reshape(1,1)
    alpha = np.array(0).reshape(1,1)
    
    err = np.append(err,d[0] - h.T @ alpha)
    for n in range(1, len(d)):
        u_n = u[n].reshape(1,2)
        d_n = d[n]

        if np.max(np.abs([kernel(u_n,dictionary[j]) for j in range(len(dictionary))])) < threshold:
            m += 1
            dictionary = np.r_[dictionary, u_n]
            h = np.array([kernel(u_n,dictionary[j]) for j in range(len(dictionary))]).T.reshape(m,1)
            alpha = np.r_[alpha,[[0]]]

        h = np.array([kernel(u_n,dictionary[j]) for j in range(len(dictionary))]).T.reshape(m,1)
        err = np.append(err,d_n - h.T @ alpha)
        alpha = alpha + (step_size)*((d_n - h.T @ alpha).item() * h)


#     print('number of SVs',len(dictionary))
    return err