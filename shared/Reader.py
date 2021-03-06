from random import shuffle, seed
import numpy as np
import os

root = os.path.abspath(os.path.dirname(__file__))
CHIPS_PATH = os.path.join(root, 'chips.txt')

def read_training_set(shuf=True, rseed=20):
    seed(rseed)

    objects = []

    with open(CHIPS_PATH, 'r') as file:
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
    Y = np.array(Y)

    return X, Y
