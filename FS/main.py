import os
from numpy import genfromtxt
import numpy as np
from FS.metrics.spearman import *
from kNN.kNNClassifier import KNNClassifier
from collections import defaultdict
from FS.metrics.spearman import Spearman
from FS.metrics.pears import Pears
from FS.metrics.ig import IG
from FS.drawer import draw


def read():
    root = os.path.abspath(os.path.dirname(__file__))
    TRAIN_X_FILE = os.path.join(root, 'arcene_train.data')
    TRAIN_Y_FILE = os.path.join(root, 'arcene_train.labels')
    VALID_X_FILE = os.path.join(root, 'arcene_valid.data')
    VALID_Y_FILE = os.path.join(root, 'arcene_valid.labels')

    train_X = genfromtxt(TRAIN_X_FILE, delimiter=' ')
    train_Y = genfromtxt(TRAIN_Y_FILE, delimiter=' ')
    valid_X = genfromtxt(VALID_X_FILE, delimiter=' ')
    valid_Y = genfromtxt(VALID_Y_FILE, delimiter=' ')

    return train_X, train_Y, valid_X, valid_Y


def get_indices(ps, treshold):
    indices = []
    for i in range(len(ps)):
        if np.abs(ps[i]) > treshold:
            indices.append(i)
    return indices


def get_x_by_indices(old_x, indices):
    X = []
    for x in old_x:
        X.append(np.take(x, indices))

    X = np.array(X)
    return X


def write_indices(metric, treshold, indices):
    root = os.path.abspath(os.path.dirname(__file__))
    name = os.path.join(root, 'indices.txt')
    file = open(name, "a")
    file.write(str(metric) + "-" + str(treshold) + ":\n")
    file.write(str(indices) + "\n")
    file.close()


def classify(train_X, train_Y, valid_X, valid_Y):
    X = train_X
    Y = train_Y

    classifier = KNNClassifier()
    classifier.train(X, Y)

    X = valid_X
    Y = valid_Y

    F = f1_measure(X, Y, classifier)

    return F


def f1_measure(X, Y, clf):
    guessed_label = [clf.predict(sample) for sample in X]

    # -1, -1 - tn; -1, 1 - fp; 1, -1 - fn; 1, 1 - tp
    counts = defaultdict(lambda: defaultdict(lambda: 0))
    for cc, gc in zip(Y, guessed_label):
        counts[cc][gc] += 1

    if counts[1][1] == 0:
        return 0

    recall = counts[1][1] / (counts[1][1] + counts[1][-1])
    precision = counts[1][1] / (counts[1][1] + counts[-1][1])

    f1_measure = 2 * precision * recall / (precision + recall)
    return f1_measure


def main():
    train_X, train_Y, valid_X, valid_Y = read()
    metrics = [Spearman, Pears, IG]
    for metric_cls in metrics:
        metric = metric_cls()
        correlations = metric.get_correlations(train_X, train_Y)
        for treshold in metric.lucky_tresholds():
            indices = get_indices(correlations, treshold)
            new_train_X = get_x_by_indices(train_X, indices)
            new_valid_X = get_x_by_indices(valid_X, indices)
            F = classify(new_train_X, train_Y, new_valid_X, valid_Y)
            write_indices(metric, treshold, indices)
            print(str(metric) + "-" + str(treshold) + ": " + str(F))


# Spearman treshold: 0.01 - 0.03
# Pears treshold:    0.1  - 0.3
# IG treshold:       100  - 200
if __name__ == "__main__":
    main()