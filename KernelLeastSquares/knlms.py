import numpy as np
import matplotlib.pyplot as plt

def KNLMS(d,d_true,kernel,step_size,reg_coeff,threshold):
    
    mse_KNLMS = np.array([])
    
    # Initialization
    m = 1
    u = np.matrix([d[0],d[1]])
    dictionary = np.matrix(u)

    h = np.matrix(kernel(u,dictionary))
    alpha = np.matrix(0)

    for n in range(1, len(d)):
        u_n = np.matrix([d[n-1],d[n]])
        d_n = np.matrix(d[n])

        if np.max(np.abs([kernel(u_n,dictionary[j]) for j in range(len(dictionary))])) < threshold:
            m += 1
            dictionary = np.r_[dictionary, u_n]
            h = np.matrix([kernel(u_n,dictionary[j]) for j in range(len(dictionary))]).T
            alpha = np.r_[alpha,[[0]]]

        h = np.matrix([kernel(u_n,dictionary[j]) for j in range(len(dictionary))]).T
        alpha = alpha + (step_size/(reg_coeff + (np.linalg.norm(h,ord=2)**2)))*((d_n - h.T @ alpha).item() * h)
        mse_KNLMS = np.append(mse_KNLMS, (d_true[n]-d_n + (d_n - h.T @ alpha).item())**2)

    mse_KNLMS_smooth = np.convolve(mse_KNLMS,np.ones(20),'valid') / 20
    plt.semilogy(range(len(mse_KNLMS_smooth)),mse_KNLMS_smooth)