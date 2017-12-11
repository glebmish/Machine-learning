import os
from numpy import genfromtxt
import numpy as np
from FS.spearman.spearman import *

def main():
    count_treshold = 0.03

    root = os.path.abspath(os.path.dirname(__file__))
    TRAIN_X_FILE = os.path.join(root, 'arcene_train.data')
    TRAIN_Y_FILE = os.path.join(root, 'arcene_train.labels')

    train_X = genfromtxt(TRAIN_X_FILE, delimiter=' ')
    train_Y = genfromtxt(TRAIN_Y_FILE, delimiter=' ')

    indices = spearman(train_X, train_Y, count_treshold)

    X = []
    for x in train_X:
        X.append(np.take(x, indices))

    X = np.array(X)


if __name__ == "__main__":
    main()