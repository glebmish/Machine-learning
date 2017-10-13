import Point


class Base:
    def __init__(self, n, normal):
        self.n = n
        self.normal = normal

    def metric(self, point):
        return 0
