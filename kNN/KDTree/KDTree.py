from kNN.KDTree.Division import Division
from kNN.KDTree.KDNode import KDNode
from kNN.reader import *


class KDTree:

    def __init__(self, p):
        self.root = self.build(p, 0)

    # returns KDNode
    @staticmethod
    def build(p, depth):
        if len(p) == 1:
            return KDNode(p[0], None, None, None, None)
        elif depth / 2 == 0:
            division = Division.x
        else:
            division = Division.y

        p1, p2, median = KDTree.divide(p, division)

        if len(p1) == 0:
            p1, p2, tmp = KDTree.divide(p2, Division.switch_division(division))
        elif len(p2) == 0:
            p1, p2, tmp = KDTree.divide(p1, Division.switch_division(division))

        return KDNode(
            None,
            median,
            division,
            KDTree.build(p1, depth+1),
            KDTree.build(p2, depth+1)
        )

    # divides list of points into two lists by median, found by method
    @staticmethod
    def divide(p, div):
        if div == Division.x:
            median = 0
            for point in p:
                median += point.x
            median /= len(p)
            p1 = [point for point in p if point.x < median]
            p2 = [point for point in p if point.x >= median]
        else:
            median = 0
            for point in p:
                median += point.y
            median /= len(p)
            p1 = [point for point in p if point.y < median]
            p2 = [point for point in p if point.y >= median]

        return p1, p2, median


if __name__ == '__main__':
    objects = read_training_set()
    kd = KDTree(objects)
    # still have no idea how to use it. Seems like we should
    # traverse down the tree and count distance each time to
    # decide if we should go left or right
