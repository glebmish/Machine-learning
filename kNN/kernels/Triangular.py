import math

from kNN.kernels.KernelBase import KernelBase


class Triangular(KernelBase):

    def function(self, u):
        return 1 - math.fabs(u)

    def __str__(self):
        return "Triangular"

    def __repr__(self):
        return self.__str__()
