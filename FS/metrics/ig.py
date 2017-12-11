import numpy as np
from FS.metrics.base import Base


class IG(Base):

    def __init__(self, treshold):
        super().__init__(treshold)

    @staticmethod
    def ig(X, Y, s_Y):
        S = IG.count_sum(X)
        S += s_Y


    @staticmethod
    def count_sum(X):
        S = 0
        unique, counts = np.unique(X, return_counts=True)
        counter = dict(zip(unique, counts))
        for x in X:
            p = counter[x] / len(X)
            S += p * np.log2(p)
        return S


    def get_correlation_indices(self, train_X, train_Y):
        attribute_indices = []

        s_Y = IG.count_sum(train_Y)

        for i in range(train_X.shape[1]):
            p = IG.ig(train_X[:, 1], train_Y, s_Y)
            if np.abs(p) > self.treshold:
                attribute_indices.append(i)

        print(str(len(attribute_indices)) + " attributes passed treshold")

        return attribute_indices