import math

from kNN.kernels.KernelBase import KernelBase


class Sigmoid(KernelBase):
    def __init__(self, metric):
        super().__init__(metric)

    def function(self, u):
        return 2 / (math.pi * (math.pow(math.e, u) + math.pow(math.e, -u)))
