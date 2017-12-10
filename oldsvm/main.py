import SupportVector.SVM as SVM
import numpy as np

import shared.Reader as reader
from kNN.Point import Point
from kNN.Visualizer import Visualizer

if __name__ == "__main__":

    X, y = reader.read_training_set(shuf=True)
    split = round(len(X) * 0.9)

    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]

    svm = SVM.SVM(kernel=SVM.polynomial_kernel)
    svm.fit(X_train, y_train)

    y_predict = svm.predict(X_test)
    correct = np.sum(y_predict == y_test)
    print("{} out of {} predictions correct".format(correct, len(y_predict)))

    train_set = []
    for pt, res in zip(X_train, y_train):
        train_set.append(Point(pt[0], pt[1], res))

    test_set = []
    for pt, res in zip(X_test, y_test):
        test_set.append(Point(pt[0], pt[1], res))

    clas_set = []
    for pt, res in zip(X_test, y_predict):
        clas_set.append(Point(pt[0], pt[1], res))

    Visualizer().visualize(train_set, test_set, clas_set)