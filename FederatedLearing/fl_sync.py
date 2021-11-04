import numpy as np

def fl_sync(iteration,K,kernel,h,u,d,l):
    D = len(h)
    c =  np.ones(K).reshape(K,1)/K    
    mse = np.zeros(iteration)
    alpha = np.zeros((D,1))  
    alphas = np.zeros((K,D))
    
    mse[0] = np.var(d)

    for n in range(iteration):

        # Local updates
        v = np.random.randint(0,len(d))
        edge = np.random.randint(0,K)
        alpha_in = alphas[edge].reshape((D,1))
        u_k = u[v]
        h_k = h[:,v].reshape((D,1))
        d_k = np.array([d[v]])
        _,alpha_step = kernel.train(h_k,d_k,alpha_in)

        alphas[edge] = alpha_step.T

        if n % l == 0 and n>0:
            alpha = (alphas.T @ c)
            alphas = np.repeat(alpha,K,axis=1).T
            mse[n] = np.square(np.linalg.norm(d[-500::].reshape(500,1) - h.T[-500::] @ alpha))/500
        elif n > 0:

            mse[n] = mse[n-1]        
    return mse