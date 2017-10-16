class Point(object):

    def __init__(self, x, y, cls=None):
        self.x = x
        self.y = y
        self.cls = cls

    def __str__(self):
        return "Object [x={}, y={}, cls={}]".format(self.x, self.y, self.cls)

    def __repr__(self):
        return self.__str__()

    @staticmethod
    def cmp_x(point_a, point_b):
        return point_a.x - point_b.x

    @staticmethod
    def cmp_y(point_a, point_b):
        return point_a.y - point_b.y