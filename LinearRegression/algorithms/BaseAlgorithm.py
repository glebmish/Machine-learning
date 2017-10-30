import numpy as np


class BaseAlgorithm(object):

    def __init__(self):
        self.W = None
        self.y_max = None
        self.normalize = None

    def fit(self):
        return None

    def predict(self, data_X):
        X = np.zeros(data_X.shape)
        if self.normalize:
            X[:, 0] = data_X[:, 0] / np.max(data_X[:, 0])
            X[:, 1] = data_X[:, 1] / np.max(data_X[:, 1])
            return (self.W[0] + self.W[1] * X[:, 0] + self.W[2] * X[:, 1]) * self.y_max
        else:
            return self.W[0] + self.W[1] * data_X[:, 0] + self.W[2] * data_X[:, 1]

    @staticmethod
    def mean_deviation(real, predicted):
        return np.sqrt((np.sum((real - predicted) ** 2)) / real.shape[0])