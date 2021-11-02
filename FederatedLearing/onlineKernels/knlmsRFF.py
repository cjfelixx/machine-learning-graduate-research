import numpy as np

class KNLMS_RFF:
    
    def __init__(self,d,h,step_size,reg_coeff,D,alpha_0):
        self.d = d
        self.h = h
        self.step_size = step_size
        self.reg_coeff = reg_coeff
        self.D = D
        self.alpha_0 = alpha_0
     
    def train(self):
        alpha = self.alpha_0
        h = self.h
        step_size = self.step_size
        reg_coeff = self.reg_coeff
        d = self.d
        D = self.D

        err = []
        for n in range(len(d)):
            h_n = h.T[n].reshape((D,1))
            err.append((d[n] - h_n.T @ alpha).item()) 
            alpha = alpha + (step_size/(reg_coeff + (h_n.T @ h_n))) * (err[-1] * h_n)
        return err,alpha