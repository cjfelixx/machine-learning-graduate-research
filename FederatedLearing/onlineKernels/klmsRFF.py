import numpy as np

class klms_rff:
    
    def __init__(self,step_size,D):
        self.step_size = step_size
        self.D = D
     
    def train(self,h,d,alpha_0):
        step_size = self.step_size
        D = self.D
        alpha = alpha_0
        err = []
        for n in range(len(d)):
            h_n = h.T[n].reshape((D,1))
            err.append((d[n] - h_n.T @ alpha).item()) 
            alpha += step_size * err[-1] * h_n
        return err,alpha