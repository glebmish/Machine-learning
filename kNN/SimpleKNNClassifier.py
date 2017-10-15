from kNN import metrics
from kNN import kernels

import numpy as np


class SimpleKNNClassifier(object):
    def __init__(self, k, kernel, metric):
        self.__k = k
        self.__set = None
        assert isinstance(metric, metrics.MetricBase.MetricBase)
        assert isinstance(kernel, kernels.KernelBase.KernelBase)
        self.__kernel = kernel
        self.__metric = metric

    def train(self, train_set):
        self.__set = train_set

    def classify(self, new_point):
        distances = []
        for point in self.__set:
            distances.append(self.__metric.distance(point, new_point))
        closest_distances = sorted(distances)[:self.__k]
        max_distance = np.array(closest_distances).max()

        # array of indices of points who's distances are included to closest_distances
        indices = [distances.index(distance) for distance in closest_distances]
        scores = [0, 0]
        for index in indices:
            point = self.__set[index]
            normalized_distance = distances[index] / max_distance
            kernel_value = self.__kernel.function(normalized_distance)
            scores[point.cls] += kernel_value

        return np.array(scores).argmax()
