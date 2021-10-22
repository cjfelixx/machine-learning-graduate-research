import numpy as np

def KNLMS(u,d,kernel_params,step_size,reg_coeff,threshold,alpha_0=np.array(0).reshape(1,1)):

    sigma = kernel_params.sigma
    kernel = lambda u_i,u_j: np.exp(-1 * sigma * (np.linalg.norm(u_i - u_j,ord=2)**2))  

    err = np.array([])
    
    # Initialization
    m = 1
    u_0 = u[0]
    dictionary = np.array(u_0).reshape(1,2)
    h = np.array(kernel(u_0,dictionary)).reshape(1,1)
    alpha = alpha_0
    
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
        alpha = alpha + (step_size/(reg_coeff + (np.linalg.norm(h,ord=2)**2)))*((d_n - h.T @ alpha).item() * h)


#     print('number of SVs',len(dictionary))
    return err