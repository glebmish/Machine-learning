import math

from kNN.kernels.Base import Base


class Gaussian(Base):
    def __init__(self, metric):
        super().__init__(metric)

    def function(self, u):
        return math.pow(math.e, -u * u / 2) / math.sqrt(2 * math.pi)
