class MetricBase:
    def __init__(self, n):
        self.n = n

    def distance(self, point_a, point_b):
        return 0

    def __str__(self):
        return "MetricBase " + self.n

    def __repr__(self):
        return self.__str__()
