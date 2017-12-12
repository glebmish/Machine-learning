import numpy as np
from FS.metrics.base import Base


class Pears(Base):

    def __init__(self):
        super().__init__()


    @staticmethod
    def pears(i, x, y, mean_y, variance_sqr_y):
        mean_x = np.mean(x)
        sum_up = np.sum((x - mean_x) * (y - mean_y))
        variance_sqr_x = np.sum((x - mean_x) ** 2)
        sum_down = np.sqrt(variance_sqr_x * variance_sqr_y)
        return sum_up / sum_down


    def get_correlations(self, train_X, train_Y):
        mean_y = np.mean(train_Y)
        variance_sqr_y = np.sum((train_Y - mean_y) ** 2)
        ps = []

        for i in range(train_X.shape[1]):
            p = Pears.pears(i, train_X[:, i], train_Y, mean_y, variance_sqr_y)
            ps.append(p)

        return ps


    @staticmethod
    def lucky_tresholds():
        return np.linspace(0.1, 0.3, 21)


    def __str__(self):
        return "Pears"
