from kNN.KDTree.Division import Division
from kNN.KDTree.KDNode import KDNode
from statistics import median
from kNN.reader import *


class KDTree:

    def __init__(self, p):
        self.root = self.build(p, 0)

    # returns KDNode
    @staticmethod
    def build(p, depth):
        if len(p) == 1:
            return KDNode(p[0], None, None, None, None)
        elif depth % 2 == 0:
            division = Division.x
        else:
            division = Division.y

        p1, p2, med = KDTree.divide(p, division)

        if len(p1) == 0:
            p1, p2, tmp = KDTree.divide(p2, Division.switch_division(division))
        elif len(p2) == 0:
            p1, p2, tmp = KDTree.divide(p1, Division.switch_division(division))

        left = KDTree.build(p1, depth + 1)
        right = KDTree.build(p2, depth + 1)

        return KDNode(
            None,
            med,
            division,
            left,
            right
        )

    # divides list of points into two lists by median, found by method
    @staticmethod
    def divide(p, div):
        if div == Division.x:
            p.sort(key=Division.div_x)
            med = median([point.x for point in p])
            p1 = [point for point in p if point.x < med]
            p2 = [point for point in p if point.x >= med]
            print("X median is: ", med)
            print(p1)
            print(p2)
        else:
            p.sort(key=Division.div_y)
            med = median([point.y for point in p])
            p1 = [point for point in p if point.y < med]
            p2 = [point for point in p if point.y >= med]
            print("Y median is: ", med)
            print(p1)
            print(p2)

        return p1, p2, med


if __name__ == '__main__':
    objects = read_training_set()
    kd = KDTree(objects)
    print("Done")
    # still have no idea how to use it. Seems like we should
    # traverse down the tree and count distance each time to
    # decide if we should go left or right
