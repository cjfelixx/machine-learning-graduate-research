import numpy as np

class KLMS_RFF:
    
    def __init__(self,d,h,step_size,D,alpha_0):
        self.d = d
        self.h = h
        self.step_size = step_size
        self.D = D
        self.alpha_0 = alpha_0
     
    def train(self):
        alpha = self.alpha_0
        h = self.h
        step_size = self.step_size
        d = self.d
        D = self.D

        err = []
        for n in range(len(d)):
            h_n = h.T[n].reshape((D,1))
            err.append((d[n] - h_n.T @ alpha).item()) 
            alpha += step_size * err[-1] * h_n
        return err,alpha