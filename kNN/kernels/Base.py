import metrics


class Base:
    def __init__(self, metric):
        assert isinstance(metric, metrics.Base.Base)
        self.metric = metric

    def function(self, u):
        return 1 / 2
