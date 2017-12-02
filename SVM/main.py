from random import shuffle

import reader
import SVM
import numpy as np


def plot_margin(X_train, y_train, clf):
    def f(x, w, b, c=0):
        # given x, return y such that [x,y] in on the line
        # w.x + b = c
        return (-w[0] * x - b + c) / w[1]

    X1_train = []
    X2_train = []

    for eachX, eachy in zip(X_train, y_train):
        if eachy == 1:
            X1_train.append(eachX)
        else:
            X2_train.append(eachX)

    pl.plot(X1_train[:, 0], X1_train[:, 1], "ro")
    pl.plot(X2_train[:, 0], X2_train[:, 1], "bo")
    pl.scatter(clf.sv[:, 0], clf.sv[:, 1], s=100, c="g")

    # w.x + b = 0
    a0 = -4;
    a1 = f(a0, clf.w, clf.b)
    b0 = 4;
    b1 = f(b0, clf.w, clf.b)
    pl.plot([a0, b0], [a1, b1], "k")

    # w.x + b = 1
    a0 = -4;
    a1 = f(a0, clf.w, clf.b, 1)
    b0 = 4;
    b1 = f(b0, clf.w, clf.b, 1)
    pl.plot([a0, b0], [a1, b1], "k--")

    # w.x + b = -1
    a0 = -4;
    a1 = f(a0, clf.w, clf.b, -1)
    b0 = 4;
    b1 = f(b0, clf.w, clf.b, -1)
    pl.plot([a0, b0], [a1, b1], "k--")

    pl.axis("tight")
    pl.show()

if __name__ == "__main__":
    data = reader.read_training_set()
    shuffle(data)

    svm = SVM.SVM()

    train_size = round(len(data) * 0.8)

    train_set = data[:train_size]
    test_set = data[train_size:]

    X = np.array([[point.x, point.y] for point in train_set])
    y = np.array([point.cls for point in train_set])

    svm.train(X, y)
    X_train = [[point.x, point.y] for point in test_set]
    y_train = []

    for each in X:
        y_train.append(svm.classify(each))

    plot_margin(X_train[y_train == 1], X_train[y_train == -1], clf)
