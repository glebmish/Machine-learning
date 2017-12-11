import math
from kNN.Point import Point
from kNN.transformers.TransformerBase import TransformerBase


class Rad(TransformerBase):
    def transform(self, X):
        new_set = []
        for x in X:
            new_set.append(Point((point.x ** 2 + point.y ** 2), math.atan2(point.x, point.y), point.cls))
        return new_set

    def __str__(self):
        return "Rad"

    def __repr__(self):
        return self.__str__()
