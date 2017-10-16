from random import seed, random
from collections import namedtuple
from math import sqrt
from copy import deepcopy

from kNN.reader import read_training_set
from kNN.Point import Point
from kNN.metrics.MetricBase import MetricBase
from kNN.metrics.Minkovsky import Minkovsky


def sqd(p1, p2):
    return sum((c1 - c2) ** 2 for c1, c2 in zip(p1, p2))


class KdNode(object):
    __slots__ = ("dom_value", "split", "left", "right")

    def __init__(self, dom_value, split, left, right):
        self.dom_value = dom_value
        self.split = split
        self.left = left
        self.right = right


class Orthotope(object):
    __slots__ = ("min", "max")

    def __init__(self, mi, ma):
        self.min, self.max = mi, ma


class KdTree(object):
    __slots__ = ("root", "bounds", "metric")

    def __init__(self, pts, bounds, metric):
        def build(split, existing_set):
            if not existing_set:
                return None
            if split % 2 == 0:
                existing_set = sorted(existing_set, key=lambda point: point.x)
            else:
                existing_set = sorted(existing_set, key=lambda point: point.y)
            half_count = len(existing_set) // 2
            middle = existing_set[half_count]
            while half_count + 1 < len(existing_set) and \
                    (split % 2 == 0 and existing_set[half_count + 1].x == middle.x or
                     split % 2 != 0 and existing_set[half_count + 1].y == middle.y):
                half_count += 1

            s2 = (split + 1) % 2  # cycle coordinates
            return KdNode(middle, split, build(s2, existing_set[:half_count]),
                          build(s2, existing_set[half_count + 1:]))

        assert isinstance(metric, MetricBase)
        self.root = build(0, pts)
        self.bounds = bounds
        self.metric = metric

    def find_nearest(self, k, p):
        def find(kd, target, limit, max_dist_sqd):
            T3 = namedtuple("T3", "nearest dist_sqd nodes_visited")

            if kd is None:
                return T3([Point(0.0, 0.0)] * k, float("inf"), 0)

            nodes_visited = 1
            s = kd.split
            pivot = kd.dom_value
            left_limit = deepcopy(limit)
            right_limit = deepcopy(limit)

            if s % 2 == 0:
                pivot_s = pivot.x
                target_s = target.x
                left_limit.max.x = pivot.x
                right_limit.min.x = pivot.x
            else:
                pivot_s = pivot.y
                target_s = target.y
                left_limit.max.y = pivot.y
                right_limit.min.y = pivot.y

            if target_s <= pivot_s:
                nearer_kd, nearer_hr = kd.left, left_limit
                further_kd, further_hr = kd.right, right_limit
            else:
                nearer_kd, nearer_hr = kd.right, right_limit
                further_kd, further_hr = kd.left, left_limit

            n1 = find(nearer_kd, target, nearer_hr, max_dist_sqd)
            nearest = n1.nearest
            dist_sqd = n1.dist_sqd
            nodes_visited += n1.nodes_visited

            if dist_sqd < max_dist_sqd:
                max_dist_sqd = dist_sqd
            d = (pivot_s - target_s) ** 2
            if d > max_dist_sqd:
                return T3(nearest, dist_sqd, nodes_visited)
            d = self.metric.distance(pivot, target)
            if d < dist_sqd:
                nearest = pivot
                dist_sqd = d
                max_dist_sqd = dist_sqd

            n2 = find(further_kd, target, further_hr, max_dist_sqd)
            nodes_visited += n2.nodes_visited
            if n2.dist_sqd < dist_sqd:
                nearest = n2.nearest
                dist_sqd = n2.dist_sqd

            return T3(nearest, dist_sqd, nodes_visited)

        return find(self.root, p, self.bounds, float("inf"))


def show_nearest(k, heading, kd, p):
    print(heading + ":")
    print("Point:           ", p)
    n = kd.find_nearest(k, p)
    print("Nearest neighbor:", n.nearest)
    print("Distance:        ", sqrt(n.dist_sqd))
    print("Nodes visited:   ", n.nodes_visited, "\n")


def random_point(k):
    return [random() for _ in range(k)]


def random_points(k, n):
    return [random_point(k) for _ in range(n)]


if __name__ == "__main__":
    points = read_training_set()
    target = Point(0.5, 0.5)
    metric = Minkovsky(2)
    print(sorted([metric.distance(target, point) for point in points]))
    kd = KdTree(points, Orthotope(Point(-2.0, -2.0), Point(2.0, 2.0)), metric)
    show_nearest(2, "Example", kd, target)
