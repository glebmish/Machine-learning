from random import shuffle

import reader
import SupportVector
import numpy as np
import pylab as pl


def plot_margin(X1_train, X2_train, clf):
    def f(x, w, b, c=0):
        # given x, return y such that [x,y] in on the line
        # w.x + b = c
        return (-w[0] * x - b + c) / w[1]

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

    train_size = round(len(data) * 0.8)

    train_set = data[:train_size]
    test_set = data[train_size:]

    X_train = np.array([[point.x, point.y] for point in train_set])
    y_train = np.array([point.cls for point in train_set])

    X_test = np.array([[point.x, point.y] for point in test_set])
    y_test = np.array([point.cls for point in test_set])

    clf = SupportVector.SVM(C=0.1)
    clf.fit(X_train, y_train)

    y_predict = clf.predict(X_test)
    correct = np.sum(y_predict == y_test)
    print("{} out of {} predictions correct".format(correct, len(y_predict)))

    plot_margin(X_train[y_train == 1], X_train[y_train == -1], clf)
