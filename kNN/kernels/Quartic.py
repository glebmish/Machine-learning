from kNN.kernels.KernelBase import KernelBase


class Quartic(KernelBase):

    def function(self, u):
        return 15 * ((1 - u ** 2) ** 2) / 16

    def __str__(self):
        return "Quartic"

    def __repr__(self):
        return self.__str__()
