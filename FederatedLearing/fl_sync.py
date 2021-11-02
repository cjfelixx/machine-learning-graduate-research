import numpy as np

def fl_sync(iteration,K,h,u,d):
    
    mse_cent = [np.var(d)]
    sigma = 1/np.sqrt(2*kernel.sigma)
    W = (1/sigma) * np.random.normal(size=(2,D))

    b = np.random.uniform(0,np.pi,(D,1))
    h = np.sqrt(2/D) * np.cos(W.T @ u.T + b)
    for n in tqdm(range(iteration)):

        # Local updates
        v = np.random.randint(0,num_data)
        alpha_in = alpha
        u_k = u[v]
        h_k = h[:,v].reshape((D,1))
        d_k = np.array([d[v]])
    #     err = d_k - h_k.T @ alpha_in
    #     alpha_step = alpha_in + step_size  * h_k * err
        erri,alpha_step = KLMS_RFF(u_k,d_k,h_k,step_size,D,alpha_0=alpha_in)

        alpha = alpha_step
        mse_cent.append(np.square(np.linalg.norm(d[-500::].reshape(500,1) - h.T[-500::] @ alpha))/500)

return mse_cent