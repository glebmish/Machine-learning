import numpy as np
from FS.metrics.base import Base


class IG(Base):

    def __init__(self, treshold):
        super().__init__(treshold)

    @staticmethod
    def ig():
        pass


    def get_correlation_indices(self, train_X, train_Y):
        attribute_indices = []

        for i in range(train_X.shape[1]):
            p = IG.ig(ranked_X, indices_X, ranked_Y, indices_Y, delta)
            if np.abs(p) > self.treshold:
                attribute_indices.append(i)

        print(str(len(attribute_indices)) + " attributes passed treshold")

        return attribute_indices