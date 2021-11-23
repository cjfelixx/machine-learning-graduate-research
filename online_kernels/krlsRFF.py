import numpy as np

class krls_rff:
    
    def __init__(self,D,beta=1.0,l=1.0):
        self.D = int(D)
        self.beta = beta
        self.l = l
        self.P = np.eye(D)/l
     
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
            a = P @ h_n
            k = (a)/(beta + h_n.T @ a)
            P = (P - (k @ a.T))/beta
            alpha += k * err[-1]
            
        return err,alpha,P