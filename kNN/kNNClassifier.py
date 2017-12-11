from kNN.metrics import Minkovsky
from kNN.kernels import Gaussian, Sigmoid, Triangular, Epanechnikov, Quartic

import numpy as np


class KNNClassifier(object):
    def __init__(self, k=10, kernel=Gaussian.Gaussian(), metric=Minkovsky.Minkovsky(2)):
        self.__k = k
        self.__kernel = kernel
        self.__metric = metric

        self.__X = []
        self.__y = []

    def train(self, X, y):
        self.__X = X
        self.__y = y

    def predict(self, x):
        if len(x.shape) != 1:
            x = x.reshape(x.shape[1])

        distances = []
        for sample in self.__X:
            distances.append(self.__metric.distance(sample, x))
        closest_distances = sorted(distances)[:self.__k]
        max_distance = np.array(closest_distances).max()

        # array of indices of points who's distances are included to closest_distances
        indices = [distances.index(distance) for distance in closest_distances]
        result = 0
        for index in indices:
            label = self.__y[index]

            normalized_distance = distances[index] / max_distance
            kernel_value = self.__kernel.function(normalized_distance)

            result += kernel_value * label

        return np.sign(result).item()
