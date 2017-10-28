class Flat(object):

    def __init__(self, area, rooms, price):
        self.area = area
        self.rooms = rooms
        self.price = price

    def __str__(self):
        return "Room [area={}, rooms={}, price={}]".format(self.area, self.rooms, self.price)

    def __repr__(self):
        return self.__str__()