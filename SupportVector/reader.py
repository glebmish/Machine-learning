from random import shuffle
import numpy as np
import os

root = os.path.abspath(os.path.dirname(__file__))
TRAINING_SET_FILE = os.path.join(root, 'chips.txt')


def read_training_set(shuf=True):
    objects = []

    with open(TRAINING_SET_FILE, 'r') as file:
        for line in file:
            x, y, cls = line.split(',')
            x = float(x)
            y = float(y)
            cls = 1.0 if int(cls) == 1 else -1.0

            objects.append([x, y, cls])

    if shuf:
        shuffle(objects)

    X = []
    Y = []
    for obj in objects:
        X.append([obj[0], obj[1]])
        Y.append(obj[2])

    X = np.array(X)
    y = np.array(y)

    return X, Y
