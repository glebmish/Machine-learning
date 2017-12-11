import os
from numpy import genfromtxt
import numpy as np
from FS.spearman.spearman import *
from kNN.kNNClassifier import KNNClassifier
from collections import defaultdict


def main():
    count_treshold = 0.0

    root = os.path.abspath(os.path.dirname(__file__))
    TRAIN_X_FILE = os.path.join(root, 'arcene_train.data')
    TRAIN_Y_FILE = os.path.join(root, 'arcene_train.labels')
    VALID_X_FILE = os.path.join(root, 'arcene_valid.data')
    VALID_Y_FILE = os.path.join(root, 'arcene_valid.labels')

    train_X = genfromtxt(TRAIN_X_FILE, delimiter=' ')
    train_Y = genfromtxt(TRAIN_Y_FILE, delimiter=' ')
    valid_X = genfromtxt(VALID_X_FILE, delimiter=' ')
    valid_Y = genfromtxt(VALID_Y_FILE, delimiter=' ')

    indices = spearman(train_X, train_Y, count_treshold)

    X = []
    for x in train_X:
        X.append(np.take(x, indices))

    X = np.array(X)
    Y = train_Y

    classifier = KNNClassifier()
    classifier.train(X, Y)

    X = []
    for x in valid_X:
        X.append(np.take(x, indices))
    X = np.array(X)
    Y = valid_Y

    print("F: " + str(f1_measure(X, Y, classifier)))

    """
    count_good = 0
    count_bad = 0
    for i in range(len(X)):
        if classifier.predict(X[i]) == Y[i]:
            count_good += 1
        else:
            count_bad += 1
    print("Good: " + str(count_good) + "\nBad: " + str(count_bad))
    """



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


if __name__ == "__main__":
    main()