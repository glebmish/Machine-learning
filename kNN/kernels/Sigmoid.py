import math

from kNN.kernels.KernelBase import KernelBase


class Sigmoid(KernelBase):

    def function(self, u):
        return 2 / (math.pi * (math.pow(math.e, u) + math.pow(math.e, -u)))

    def __str__(self):
        return "Sigmoid"

    def __repr__(self):
        return self.__str__()
