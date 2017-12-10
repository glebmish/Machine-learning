from collections import defaultdict
import numpy as np


def correct_percentage(X, y, clf):
    guessed_label = sum([1 for sample, label in zip(X, y) if clf.predict(sample) == label])
    return guessed_label / len(X)


def f1_measure(X, y, clf):
    guessed_label = [clf.predict(np.array(sample).reshape(1, 2)) for sample in X]

    # -1, -1 - tn; -1, 1 - fp; 1, -1 - fn; 1, 1 - tp
    counts = defaultdict(lambda: defaultdict(lambda: 0))
    for cc, gc in zip(y, guessed_label):
        counts[cc][gc] += 1

    if counts[1][1] == 0:
        return 0

    recall = counts[1][1] / (counts[1][1] + counts[1][-1])
    precision = counts[1][1] / (counts[1][1] + counts[-1][1])

    f1_measure = 2 * precision * recall / (precision + recall)
    return f1_measure


def confusion_matrix(X, y, clf):
    guessed_label = [clf.predict(np.array(sample).reshape(1, 2)) for sample in X]

    # -1, -1 - tn; -1, 1 - fp; 1, -1 - fn; 1, 1 - tp
    counts = defaultdict(lambda: defaultdict(lambda: 0))
    for cc, gc in zip(y, guessed_label):
        counts[cc][gc] += 1

    matrix = [["p\r", "1", "-1"],
              [  "1",  "",   ""],
              [ "-1",  "",   ""]]

    matrix[1][1] = counts[1][1]
    matrix[1][2] = counts[1][-1]
    matrix[2][1] = counts[-1][1]
    matrix[2][2] = counts[-1][-1]

    return np.array(matrix)
