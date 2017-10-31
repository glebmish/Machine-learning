import numpy as np


class LinRegression(object):

    def __init__(self, regression_method):
        self.W = None
        self.__regression = regression_method

    def fit(self, train_set):
        X = np.array([(x.area, x.rooms) for x in train_set])
        y = np.array([x.price for x in train_set])

        self.W = self.__regression.fit(X, y)

    def predict(self, test_set):
        X = np.array([[flat.area, flat.rooms] for flat in test_set])

        return self.__regression.predict(X)

    def mean_deviation(self, y_real, y_pred):
        return self.__regression.mean_deviation(y_real, y_pred)