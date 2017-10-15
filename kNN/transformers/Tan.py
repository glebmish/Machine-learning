import math
from kNN.Point import Point
from kNN.transformers.TransformerBase import TransformerBase


class Tan(TransformerBase):
    def transform(self, points):
        new_set = []
        for point in points:
            new_set.append(Point(math.tan(point.x), math.tan(point.y), point.cls))
        return new_set

    def __str__(self):
        return "Tan"

    def __repr__(self):
        return self.__str__()
