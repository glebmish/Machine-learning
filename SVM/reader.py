from kNN.Point import Point
import os

root = os.path.abspath(os.path.dirname(__file__))
TRAINING_SET_FILE = os.path.join(root, 'chips.txt')


def read_training_set():
    with open(TRAINING_SET_FILE, 'r') as file:
        objects = []

        for line in file:
            x, y, cls = line.split(',')
            x = float(x)
            y = float(y)
            cls = 1.0 if int(cls) == 1 else -1.0

            objects.append(Point(x, y, cls))

    return objects
