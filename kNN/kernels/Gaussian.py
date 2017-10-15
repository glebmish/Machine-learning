import math

from kNN.kernels.KernelBase import KernelBase


class Gaussian(KernelBase):

    def function(self, u):
        return math.pow(math.e, -u * u / 2) / math.sqrt(2 * math.pi)

    def __str__(self):
        return "Gaussian"

    def __repr__(self):
        return self.__str__()
