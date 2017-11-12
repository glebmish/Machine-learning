import numpy as np


class Base(object):

    def __init__(self):
        self.W = None
        self.y_max = None
        self.x_max = None
        self.normalize = None

    def fit(self, data_X, data_Y):
        return None

    def predict(self, data_X):
        X = np.ones([data_X.shape[0], data_X.shape[1] + 1])
        if self.normalize:
            X[:, 1] = data_X[:, 0] / np.max(data_X[:, 0])
            X[:, 2] = data_X[:, 1] / np.max(data_X[:, 1])
            return np.dot(X, self.W) * self.y_max
        else:
            X[:, 1] = data_X[:, 0]
            X[:, 2] = data_X[:, 1]
            return np.dot(X, self.W)

    def predict_single(self, x):
        X = np.ones(x.shape[0] + 1)
        if self.normalize:
            X[1] = x[0] / self.x_max[0]
            X[2] = x[1] / self.x_max[1]
            return np.dot(X, self.W) * self.y_max
        else:
            X[1] = x[0]
            X[2] = x[1]
            return np.dot(X, self.W)

    @staticmethod
    def mean_deviation(real, predicted):
        return np.sqrt((np.sum((real - predicted) ** 2)) / real.shape[0])