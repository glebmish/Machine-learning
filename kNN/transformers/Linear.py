from kNN.Point import Point
from kNN.transformers.TransformerBase import TransformerBase


class Linear(TransformerBase):

    def transform(self, points):
        new_set = []
        for point in points:
            new_set.append(Point(point.x, point.y, point.cls))
        return new_set

    def __str__(self):
        return "Linear"

    def __repr__(self):
        return self.__str__()
