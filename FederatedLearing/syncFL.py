import numpy as np

class fl_sync:
    def __init__(self,K,h,l):
        self.K = K
        self.h = h
        self.l = l
        
    def train(self,iteration,kernel,u,d):
        
        K = self.K
        h = self.h
        l = self.l
        D = len(h)
        c =  np.ones(K).reshape(K,1)/K    
        mse = np.zeros(iteration)
        alpha = np.zeros((D,1))  
        alphas = np.zeros((K,D))
        P = [kernel.P for i in range(K)]
        mse[0] = np.var(d)
        for n in range(1,iteration):
            # Local updates
            v = np.random.randint(len(d))
            edge = np.random.randint(K)
            P_k = P[edge]
            alpha_in = alphas[edge].reshape((D,1))
            u_k = u[v]
            h_k = h[:,v].reshape((D,1))
            d_k = np.array([d[v]])
            if P_k.any():
                _,alpha_step,P_k = kernel.train(h_k,d_k,alpha_in,P_k)
                P[edge] = P_k
            else:
                _,alpha_step = kernel.train(h_k,d_k,alpha_in)
                
            alphas[edge] = alpha_step.T

            if n % l == 0 and n>0:
                alpha = (alphas.T @ c)
                alphas = np.repeat(alpha,K,axis=1).T
                mse[n] = np.square(np.linalg.norm(d[-500::].reshape(500,1) - h.T[-500::] @ alpha))/500
            elif n > 0:

                mse[n] = mse[n-1]
        return mse