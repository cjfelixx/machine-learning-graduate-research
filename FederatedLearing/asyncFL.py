import numpy as np

class fl_async:
    
    def __init__(self,K,h,p):
        self.K = K
        self.h = h
        self.p = p

    def train(self,iteration,kernel,u,d):
        K = self.K
        p = self.p
        h = self.h
        D = len(h)
        c =  np.ones(K).reshape(K,1)/K    
        mse = np.zeros(iteration)
        alpha = np.zeros((D,1))  
        alphas = np.zeros((K,D))
        last_alphas = alphas.copy()
        P = [kernel.P for i in range(K)]        
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
            P_k = P[edge]   
            
            if P_k.any():
                _,alpha_step,P_k = kernel.train(h_k,d_k,alpha_in,P_k)
                P[edge] = P_k
            else:
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