from kNN.Point import Point

TRAINING_SET_FILE = './chips.txt'


def read_training_set():
    with open(TRAINING_SET_FILE, 'r') as file:
        objects = []

        for line in file:
            x, y, cls = line.split(',')
            x = float(x)
            y = float(y)
            cls = int(cls)

            objects.append(Point(x, y, cls))

    return objects
