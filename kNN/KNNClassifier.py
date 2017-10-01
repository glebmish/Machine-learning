import math


class KNNClassifier(object):

    def __init__(self, k):
        self.__k = k
        self.__set = []

    def train(self, train_set):
        self.__set = list(train_set)

    def classify(self, point):
        k_nearest = self.__find_nearest(self.__k, source=self.__set, comparator=self.__distance(point))
        return round(sum(p.cls for p in k_nearest) / len(k_nearest))

    @staticmethod
    def __distance(p1):
        return lambda p2: math.sqrt((p2.x - p1.x) ** 2 + (p2.y - p1.y) ** 2)

    @staticmethod
    def __find_nearest(k, source, comparator):
        source = source[:]
        source.sort(key=comparator)
        return source[:k]

