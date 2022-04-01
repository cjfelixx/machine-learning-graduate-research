import numpy as np
from scipy.sparse import spdiags

def W_norm(W,norm):
    N = W.shape[1]
    
    if norm == 2:
        norms = np.sqrt(np.sum(W*W,0)).T
    else:
        norms = np.sqrt(np.sum(np.abs(W),0)).T
    W *= spdiags(norms**-1,0,N,N)
    return W

# def calc_Obj(Y,T,B,W,A,index=0):
#     MAXARRAY = 500*1024*2014/8
    
#     N = Y.shape[0]    
#     B0 = np.zeros(N)
    
#     dV = []
#     mn = A.shape[0] * A.shape[1]
    
#     nBlock = np.ceil(mn/MAXARRAY)
    
#     if mn < MAXARRAY:
#         obj_NMF = 0
#         dY = []
#         for i in range(N):
#             dY.append(obj_NMF + dY[])

    
    