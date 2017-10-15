from kNN.metrics import MetricBase


class KernelBase:

    def function(self, u):
        return 1 / 2

    def __str__(self):
        return "KernelBase"

    def __repr__(self):
        return self.__str__()
