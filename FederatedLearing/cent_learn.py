import numpy as np

def cent_learn(iteration,K,kernel,h,u,d):
    D = len(h)
    mse = np.zeros(iteration)
    alpha = np.zeros((D,1))  
    mse[0] = np.var(d)

    for n in range(iteration):

        v = np.random.randint(0,len(d))
        alpha_in = alpha
        u_k = u[v]
        h_k = h[:,v].reshape((D,1))
        d_k = np.array([d[v]])
        _,alpha_step = kernel.train(h_k,d_k,alpha_in)

        alpha = alpha_step

        mse[n] = np.square(np.linalg.norm(d[-500::].reshape(500,1) - h.T[-500::] @ alpha))/500
       
    return mse