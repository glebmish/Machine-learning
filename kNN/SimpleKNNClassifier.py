from structures.Set.set import Set


class KNNClassifier(object):
    def __init__(self, k, kernel, metric):
        self.__k = k
        self.__set = None
        self.__kernel = kernel
        self.__metric = metric

    def train(self, train_set):
        self.__set = Set(train_set, self.__kernel, self.__metric)

    def classify(self, point):
        k_nearest = self.__set.find_n_neighbors(self.__k, point)
        return self.__resolve_class(k_nearest)

    def __resolve_class(self, k_nearest):
        class_1 = [p for p in k_nearest if p.cls == 1]
        class_0 = [p for p in k_nearest if p.cls == 0]
        if len(class_1) > len(class_0):
            return 1
        else:
            return 0
