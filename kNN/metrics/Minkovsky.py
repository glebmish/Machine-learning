import math

from kNN.metrics.MetricBase import MetricBase


class Minkovsky(MetricBase):
    def __init__(self, n):
        super().__init__(n)

    def distance(self, x1, x2):
        sum = 0
        for each1, each2 in zip(x1, x2):
            sum += math.pow(math.fabs((each1 - each2)), self.n)

        return math.pow(sum, 1/self.n)

    def __str__(self):
        return "Minkovsky-" + str(self.n)

    def __repr__(self):
        return self.__str__()
