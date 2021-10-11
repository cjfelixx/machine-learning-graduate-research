def KLMS(d,kernel,step_size,reg_coeff,threshold):
    
    err = np.array([])
    # Initialization
    u_0 = np.array([d[0],d[1]]).reshape(2,1)

    h = np.zeros((D,1))

    alpha = np.zeros((D,1))
    err = np.append(err,d[0] - h.T @ alpha)
    alpha = alpha + step_size * err[-1] * h
    for n in range(1, len(d)):
        u_n = np.array([d[n-1],d[n]]).reshape(2,1)
        d_n = d[n]
        h = np.sqrt(2/D) * np.cos(W.T @ u_n + b)        
        err = np.append(err, d_n - h.T @ alpha)
        alpha = alpha + step_size * err[-1] * h
    return err
