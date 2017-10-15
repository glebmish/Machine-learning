from kNN.metrics import MetricBase


class KernelBase:
    def __init__(self, metric):
        assert isinstance(metric, MetricBase.MetricBase)
        self.metric = metric

    def function(self, u):
        return 1 / 2
