import numpy as np

class KLMS_RFF:
    
    def __init__(self,d,h,D,alpha_0,beta=1.0,l=1.0):
        self.d = d
        self.h = h
        self.D = D
        self.alpha_0 = alpha_0
        self.beta = beta
        self.l = l
     
    def train(self):
        d = self.d
        alpha = self.alpha_0
        h = self.h
        D = self.D
        beta = self.beta
        l = self.l

        err = []
        P = np.eye(D)/l
        alpha = alpha_0
        for n in range(len(d)):
            h_n = h.T[n].reshape((D,1))
            err.append((d[n] - h_n.T @ alpha).item())  
            a = h_n.T @ P
            k = (P @ h_n)/(beta + a @ h_n)
            P = (P - (k @ a))/beta
            alpha = alpha + k * err[-1]
            
        return err,alpha