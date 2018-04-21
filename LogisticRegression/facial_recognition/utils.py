import numpy as np
import pandas as pd

def init_weight_and_bias(M1, M2):
    weight = np.random.randn(M1, M2) / np.sqrt(M1 + M2)
    bias = np.zeros(M2)
    return weight.astype(np.float32), bias.astype(np.float32)

def init_filter(shape, poolsz):
    weight = np.random.randn(*shape) / np.sqrt(np.prod(shape[1:]) + shape[0]*np.prod(shape[2:] / np.prod(poolsz)))
    return weight.astype(np.float32)

def relu(x):
    return x * (x > 0)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    expx = np.exp(x)
    return expx / expx.sum(axis=1, keepdims=True)

def sigmoid_cost(T, Y):
    return -(T*np.log(Y) + (1-T)*np.log(1-Y)).sum()

def cost(T, Y):
    return -(T*np.log(Y)).sum()

def cost2(T, Y):
    N = len(T)
    return -np.log(Y[np.arange(N), T]).sum()

def error_rate(targets, preds):
    return np.mean(targets != preds)

def y2indicator(y):
    N = len(y)
    K = len(set(y))
    ind = np.zeros((N, K))
    for i in range(N):
        ind[i, y[i]] = 1
    return ind

def getData(balance_ones=True):
    X = []
    Y = []
    first = True
    for line in open('fer2013/fer2013.csv'):
        if first:
            first = False
        else:
            row = line.split(',')
            Y.append(int(row[0]))
            X.append([int(p) for p in row[1].split()])
            
    X, Y = np.array(X) / 255.0, np.array(Y)
    
    if balance_ones:
        X0, Y0 = X[Y != 1, :], Y[Y != 1]
        X1 = X[Y == 1, :]
        X1 = np.repeat(X1, 9, axis=0)
        X = np.vstack([X0, X1])
        Y = np.concatenate((Y0, [1]*len(X1)))
        
    return X, Y

def getImageData():
    X, Y = getData()
    N, D = X.shape
    d = int(np.sqrt(D))
    X = X.reshape(N, 1, d, d)
    return X, Y

def getBinaryData():
    X = []
    Y = []
    first = True
    for line in open('fer2013/fer2013.csv'):
        if first:
            first = False
        else:
            row = line.split(',')
            y = int(row[0])
            if y == 0 or y == 1:
                Y.append(y)
                X.append([int(p) for p in row[1].split()])
            
    return np.array(X) / 255.0, np.array(Y)
        