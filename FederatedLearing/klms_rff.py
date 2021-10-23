import numpy as np

def KLMS_RFF(u,d,kernel_params,step_size,D,alpha_0):
    
    sigma = 1/np.sqrt(2*kernel_params.sigma)
    
    err = np.array([])
    W = np.random.normal(loc=0, scale=sigma, size=(2,D))
    b = np.random.uniform(0,2*np.pi,D).reshape(D,1)

    # Initialization
    u_0 = u[0].reshape(2,1)
    h = np.sqrt(2/D) * np.cos(W.T @ u_0 + b)
    alpha = alpha_0
    err = np.append(err,d[0] - h.T @ alpha)
    alpha = alpha + step_size * err[-1] * h
    for n in range(1, len(d)):
        u_n = u[n].reshape(2,1)
        d_n = d[n]
        h = np.sqrt(2/D) * np.cos(W.T @ u_n + b)
        err = np.append(err, d_n - h.T @ alpha)
        alpha = alpha + step_size * err[-1] * h
    return err,h,alpha