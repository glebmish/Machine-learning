from kNN.KDTree.KDTree import KDTree


class KNNClassifier(object):

    def __init__(self, k):
        self.__k = k
        self.__set = None

    def train(self, train_set):
        self.__set = KDTree(train_set)

    def classify(self, point):
        k_nearest = self.__set.find_n_neighbors(self.__k, point)
        return round(sum(p.cls for p in k_nearest) / len(k_nearest))
