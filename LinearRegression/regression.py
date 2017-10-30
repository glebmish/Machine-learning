import numpy as np


class LinRegression(object):

    def __init__(self, fit_method):
        self.W = None
        self.__fit = fit_method

    def fit(self, train_set):
        X = np.array([(x.area, x.rooms) for x in train_set])
        y = np.array([x.price for x in train_set])

        self.W = self.__fit(X, y)

    def predict(self, test_set):
        X = np.array([[1, flat.area, flat.rooms] for flat in test_set])
        y = np.dot(X, self.W)

        return np.ndarray.flatten(y)
