from collections import defaultdict


def correct_percentage(X, y, clf):
    guessed_label = sum([1 for sample, label in zip(X, y) if clf.predict(sample) == label])
    return guessed_label / len(X)


def f1_measure(X, y, clf):
    guessed_label = [clf.predict(sample) for sample in X]

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
