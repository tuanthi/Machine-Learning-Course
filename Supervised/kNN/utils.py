import numpy as np
import pandas as pd

def get_data(limit=None):
    print('Processing data')
    data = pd.read_csv('../../mnist.csv').as_matrix()
    np.random.shuffle(data)
    X = data[:, 1:] / 255.0
    Y = data[:, 0]
    if limit is not None:
        X, Y = X[:limit], Y[:limit]
    return X, Y

def get_xor():
    X = np.zeros((200, 2))
    X[:50] = np.random.random((50, 2)) / 2 + 0.5
    X[50:100] = np.random.random((50, 2)) / 2
    X[100:150] = np.random.random((50, 2)) / 2 + np.array([[0, 0.5]])
    X[150:] = np.random.random((50, 2)) / 2 + np.array([[0.5, 0]])
    Y = np.array([0]*100 + [1]*100)
    return X, Y

def get_donut():
    N = 200
    R_inner = 5
    R_outer = 10
    
    R1 = np.random.randn(int(N/2)) + R_inner
    theta1 = 2*np.pi*np.random.random(int(N/2))
    X_inner = np.concatenate([[R1*np.cos(theta1)], [R1*np.sin(theta1)]]).T
    
    R2 = np.random.randn(int(N/2)) + R_outer
    theta2 = 2*np.pi*np.random.random(int(N/2))
    X_outer = np.concatenate([[R2*np.cos(theta2)], [R2*np.sin(theta2)]]).T
    
    X = np.concatenate([X_inner, X_outer])
    Y = np.array([0]*(int(N/2)) + [1]*(int(N/2)))
    return X, Y