from enum import Enum
from statistics import median

from structures.KDTree.KDNode import KDNode


class Division(Enum):
    x = 'X'
    y = 'Y'

    def switch(self):
        if self == Division.y:
            return Division.x
        else:
            return Division.y

    @staticmethod
    def div_x(point):
        return point.x

    @staticmethod
    def div_y(point):
        return point.y


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

        if len(p1) == 0 or len(p2) == 0:
            division = division.switch()
        if len(p1) == 0:
            p1, p2, med = KDTree.divide(p2, division)
        elif len(p2) == 0:
            p1, p2, med = KDTree.divide(p1, division)

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
        else:
            p.sort(key=Division.div_y)
            med = median([point.y for point in p])
            p1 = [point for point in p if point.y < med]
            p2 = [point for point in p if point.y >= med]

        return p1, p2, med

    def find_n_neighbors(self, n, point):
        return KDTree.find(self.root, point, n)

    @staticmethod
    def find(node, point, n):
        if node.point is None:
            more_neighbors = None
            if node.div == Division.x:
                if point.x < node.med:
                    neighbors = KDTree.find(node.left, point, n)
                    if len(neighbors) < n:
                        more_neighbors = KDTree.find(node.right, point, n - len(neighbors))
                else:
                    neighbors = KDTree.find(node.right, point, n)
                    if len(neighbors) < n:
                        more_neighbors = KDTree.find(node.left, point, n - len(neighbors))
            else:
                if point.y < node.med:
                    neighbors = KDTree.find(node.left, point, n)
                    if len(neighbors) < n:
                        more_neighbors = KDTree.find(node.right, point, n - len(neighbors))
                else:
                    neighbors = KDTree.find(node.right, point, n)
                    if len(neighbors) < n:
                        more_neighbors = KDTree.find(node.left, point, n - len(neighbors))
            if more_neighbors is None:
                return neighbors
            else:
                return neighbors + more_neighbors
        else:
            return [node.point]