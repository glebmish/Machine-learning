import math

from kNN.kernels.KernelBase import KernelBase


class Triangular(KernelBase):
    def __init__(self, metric):
        super().__init__(metric)

    def function(self, u):
        return 1 - math.fabs(u)
