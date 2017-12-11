import numpy as np
from FS.metrics.base import Base


class Pears(Base):

    def __init__(self, treshold):
        super().__init__(treshold)


    @staticmethod
    def pears(i, x, y, mean_y, variance_sqr_y):
        mean_x = np.mean(x)
        sum_up = np.sum((x - mean_x) * (y - mean_y))
        variance_sqr_x = np.sum((x - mean_x) ** 2)
        sum_down = np.sqrt(variance_sqr_x * variance_sqr_y)
        print(str(i) + ": " + str(sum_up) + " / " + str(sum_down) + " = " + str(sum_up / sum_down))
        return sum_up / sum_down


    def get_correlation_indices(self, train_X, train_Y):
        mean_y = np.mean(train_Y)
        variance_sqr_y = np.sum((train_Y - mean_y) ** 2)
        attribute_indices = []

        for i in range(train_X.shape[1]):
            p = Pears.pears(i, train_X[:, i], train_Y, mean_y, variance_sqr_y)
            if np.abs(p) > self.treshold:
                attribute_indices.append(i)

        print(str(len(attribute_indices)) + " attributes passed treshold")

        return attribute_indices