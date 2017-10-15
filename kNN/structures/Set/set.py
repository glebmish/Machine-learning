import kNN


class Set(object):
    def __init__(self, points, kernel, metric):
        assert isinstance(points, list)
        assert isinstance(kernel, kNN.kernels.KernelBase.KernelBase)
        assert isinstance(metric, kNN.metrics.MetricBase.MetricBase)
        super().__init__()
        self.__points = points
        self.__kernel = kernel
        self.__metric = metric

    def find_n_neighbors(self, n, new_point):
        sorted_list = sorted(self.__points, key=lambda point: self.__kernel.function(self.__metric.distance(point, new_point)))
        return sorted_list[:n]
