import numpy as np

def load_dodd(num_data):

    d_true =[0.1 , 0.1]
    d = [d_true[0], d_true[1]]

    # Grab new data
    new_d_true = lambda x: x.append((0.8 - 0.5 * np.exp(-(x[-1]**2)))*x[-1] - (0.3 + 0.9*np.exp(-(x[-1]**2)))*x[-2] + 0.1*np.sin(np.pi*x[-1]))
    new_d = lambda x: x.append((0.8 - 0.5 * np.exp(-(x[-1]**2)))*x[-1] - (0.3 + 0.9*np.exp(-(x[-1]**2)))*x[-2] + 0.1*np.sin(np.pi*x[-1]) + np.random.normal(0.0,0.1))    
    for i in range(2,num_data+2):
        new_d_true(d_true)
        new_d(d)
    u = np.hstack((np.array(d[0:num_data]).reshape(num_data,1),np.array(d[1:num_data+1]).reshape(num_data,1)))
    
    d_true = d_true[2::]
    d = d[2::]
    return np.array(u),np.array(d),np.array(d_true)

def load_dodd1(num_data):
    # AWGN
    v = 0.1*np.random.normal(0,1,num_data+2) 

    # Initial conditions
    d_true =[0.1 , 0.1]
    d = [d_true[0], d_true[1]]

    # Grab new data
    new_d_true = lambda d: d.append((0.8 - 0.5 * np.exp(-(d[-1]**2)))*d[-1] - (0.3 + 0.9*np.exp(-(d[-1]**2)))*d[-2] + 0.1*np.sin(np.pi*d[-1]))
    for i in range(2,num_data+2):
        new_d_true(d_true)
        d.append(d_true[-1] + v[i])

    u = np.hstack((np.array(d[0:num_data]).reshape(num_data,1),np.array(d[1:num_data+1]).reshape(num_data,1)))
    d_true = d_true[2::]
    d = d[2::]
    return np.array(u), np.array(d),np.array(d_true)