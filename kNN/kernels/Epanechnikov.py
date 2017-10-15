from kNN.kernels.KernelBase import KernelBase


class Epanechnikov(KernelBase):

    def function(self, u):
        return 3 * (1 - u ** 2) / 4

    def __str__(self):
        return "Epanechnikov"

    def __repr__(self):
        return self.__str__()
