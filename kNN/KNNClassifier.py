from kNN.KDTree.KDTree import KDTree


def resolve_class_more_neighbours(point, k_nearest):
    return round(sum(p.cls for p in k_nearest) / len(k_nearest))


def resolve_class_kernel_function(point, k_nearest):
    def distance(p1, p2):
        return math.sqrt((p2.x - p1.x) ** 2 + (p2.y - p1.y) ** 2)
    
    def distances(cls):
        return [distance(point, p_from_k) for p_from_k in k_nearest if p_from_k.cls == cls]

    distances_0 = distances(0)
    normalized_0 = sum(distances_0) / len(distances_0)

    distances_1 = distances(1)
    normalized_1 = sum(distances_1) / len(distances_1)

    return 1 if normalized_1 >= normalized_0 else 0


class KNNClassifier(object):
    def __init__(self, k, class_resolver=resolve_class_more_neighbours):
        self.__k = k
        self.__set = None
        self.__resolve_class=class_resolver

    def train(self, train_set):
        self.__set = KDTree(train_set)

    def classify(self, point):
        k_nearest = self.__find_nearest(self.__k, source=self.__set, comparator=self.distance(point))
        return self.__resolve_class(point, k_nearest)
