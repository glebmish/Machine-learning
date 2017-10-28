from LinearRegression.Flat import Flat
import os

root = os.path.abspath(os.path.dirname(__file__))
TRAINING_SET_FILE = os.path.join(root, 'prices.txt')


def read_training_set():
    with open(TRAINING_SET_FILE, 'r') as file:
        objects = []

        for line in file:
            area, rooms, price = line.split(',')
            area = int(area)
            rooms = int(rooms)
            price = int(price)

            objects.append(Flat(area, rooms, price))

    return objects
