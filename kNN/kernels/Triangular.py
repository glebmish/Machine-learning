import math

from kNN.kernels.Base import Base


class Triangular(Base):
    def __init__(self, metric):
        super().__init__(metric)

    def function(self, u):
        return 1 - math.fabs(u)
