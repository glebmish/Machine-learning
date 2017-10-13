import math

from Point import Point
from kNN.metrics.Base import Base


class Minkovsky(Base):
    def __init__(self, n, normal):
        super().__init__(n, normal)

    def metric(self, point):
        assert isinstance(point, Point)
        sum = point.x ** self.n + point.y ** self.n
        return math.pow(sum, 1/n) / self.normal
