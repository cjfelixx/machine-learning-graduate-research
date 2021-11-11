import numpy as np

class krls_rff:
    
    def __init__(self,D,beta=1.0,l=1.0):
        self.D = D
        self.beta = beta
        self.l = l
        self.P = np.eye(D)/l
#         print(self.P)
     
    def train(self,h,d,alpha_0,P=None):
        D = self.D
        beta = self.beta
        l = self.l
        alpha = alpha_0
        err = []
        if (P.any()):
            P = self.P
        
        alpha = alpha_0
        for n in range(len(d)):
            h_n = h.T[n].reshape((D,1))
            err.append((d[n] - h_n.T @ alpha).item())  
            a = h_n.T @ P
            k = (P @ h_n)/(beta + a @ h_n)
            P = P/beta - (k @ a)/beta
            alpha += k * err[-1]
            
        return err,alpha,P