
class Object(object):

    def __init__(self, x, y, cls=None):
        self.x = x
        self.y = y
        self.cls = cls

    def __str__(self):
        return "Object [x={}, y={}, cls={}]".format(self.x, self.y, self.cls)

    def __repr__(self):
        return self.__str__()