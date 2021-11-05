import numpy as np

def fl_async(iteration,K,kernel,h,u,d,p):
    D = len(h)
    c =  np.ones(K).reshape(K,1)/K    
    mse = np.zeros(iteration)
    alpha = np.zeros((D,1))  
    alphas = np.zeros((K,D))
    last_alphas = alphas.copy()
    mse[0] = np.var(d)
    edge_count = [ 0 for i in range(K)]
    for n in range(iteration):

        # Local updates
        v = np.random.randint(len(d))
        edge = np.random.randint(K)
        edge_count[edge] += 1
        alpha_in = alphas[edge].reshape((D,1))
        u_k = u[v]
        h_k = h[:,v].reshape((D,1))
        d_k = np.array([d[v]])
        _,alpha_step = kernel.train(h_k,d_k,alpha_in)

        alphas[edge] = alpha_step.T

        if edge_count[edge] % p == 0 and n>0:
            alpha += (alphas[edge] - last_alphas[edge]).reshape((D,1))/p
            alphas[edge] = alpha.reshape((D,))
            last_alphas[edge] = alphas[edge]
            mse[n] = np.square(np.linalg.norm(d[-500::].reshape(500,1) - h.T[-500::] @ alpha))/500
            edge_count[edge] = 0
        elif n > 0:
            mse[n] = mse[n-1]    
        
    return mse