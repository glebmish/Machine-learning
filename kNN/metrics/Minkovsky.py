import math

from kNN.Point import Point
from kNN.metrics.MetricBase import MetricBase


class Minkovsky(MetricBase):
    def __init__(self, n, normal):
        super().__init__(n, normal)

    def distance(self, point_a, point_b):
        assert isinstance(point_a, Point)
        assert isinstance(point_b, Point)
        sum = math.pow(math.fabs((point_a.x - point_b.x)), self.n) \
            + math.pow(math.fabs((point_a.y - point_b.y)), self.n)
        return math.pow(sum, 1/self.n) / self.normal
