import numpy as np

class Kernel:
    def __init__(self,sigma=1.0):
        self.sigma = sigma
        self.fun = lambda u_i,u_j: np.exp(-sigma * (np.square(np.linalg.norm(u_i - u_j))))