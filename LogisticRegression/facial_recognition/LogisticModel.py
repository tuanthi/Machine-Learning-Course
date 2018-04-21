import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from utils import getBinaryData, sigmoid, sigmoid_cost, error_rate

class LogisticModel(object):
    def __init__(self):
        pass
    
    def fit(self, X, Y, learning_rate=10e-7, reg=0, epochs=100000, show_fig=False):
        X, Y = shuffle(X, Y)
        Xvalid, Yvalid = X[-1000:], Y[-1000:]
        X, Y = X[:-1000], Y[:-1000]
        
        N, D = X.shape
        self.weight = np.random.randn(D) / np.sqrt(D)
        self.bias = 0
        
        costs = []
        best_validation_error = 1
        for i in range(epochs):
            pY = self.forward(X)
            
            # gragient descent step
            self.weight -= learning_rate*(X.T.dot(pY - Y) + reg*self.weight)
            self.bias -= learning_rate*((pY - Y).sum() + reg*self.bias)
            
            if i%20 == 0:
                pYvalid = self.forward(Xvalid)
                c = sigmoid_cost(Yvalid, pYvalid)
                costs.append(c)
                e = error_rate(Yvalid, np.round(pYvalid))
                print('i:', i, 'cost:', c, 'error:', e)
                if e < best_validation_error:
                    best_validation_error = e
        np.savez('trained_model', weight=self.weight, bias=self.bias)
        print('best_validation_error:', best_validation_error)
        
        if show_fig:
            plt.plot(costs)
            plt.show()
    
    def forward(self, X):
        return sigmoid(X.dot(self.weight) + self.bias)
    
    def predict(self, X):
        pY = self.forward(X)
        return np.round(pY)
    
    def score(self, X, Y):
        predictions = self.predict(X)
        return 1 - error_rate(Y, predictions)
        