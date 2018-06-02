import numpy as np
from utils import get_data
from datetime import datetime
from scipy.stats import norm
from scipy.stats import multivariate_normal as mvn

class NaiveBayes(object):
    def fit(self, X, Y, smoothing=10e-3):
        N, D = X.shape
        self.gaussians = dict()
        self.priors = dict()
        labels = set(Y)
        for c in labels:
            current_x = X[Y == c]
            self.gaussians[c] = {
                'mean': current_x.mean(axis=0),
                # For naive Bayes
                #'var': current_x.mean(axis=0) + smoothing
                # For non-naive Bayes
                'cov': np.cov(current_x.T) + np.eye(D) * smoothing
            }
            self.priors[c] = float(len(Y[Y == c])) / len(Y)
            
    def predict(self, X):
        N, D = X.shape
        K = len(self.gaussians)
        P = np.zeros((N, K))
        for c, g in self.gaussians.items():
            # For naive Bayes
            #mean, var = g['mean'], g['var']
            # For non-naive Bayes
            mean, cov = g['mean'], g['cov']
            P[:, c] = mvn.logpdf(X, mean=mean, cov=cov) + np.log(self.priors[c])
        return np.argmax(P, axis=1)
        
    def score(self, X, Y):
        p = self.predict(X)
        return np.mean(p == Y)