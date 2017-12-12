from svm import *
from kernel import Kernel
import shared.Visualizer as vis
import shared.Validator as vld
import shared.Reader as reader
import shared.Splitter as splitter
import numpy as np


def k_folds(set, k):
    fold_size = round(len(set) / k - 0.5)

    folds = []
    for i in range(k):
        folds.append(list(set[fold_size * i: fold_size * (i + 1)]))

    return folds


if __name__ == "__main__":
    X, y = reader.read_training_set(shuf=True)
    k = 9

    folds_X = k_folds(X, k)
    folds_y = k_folds(y, k)

    f1s = []

    for i in range(k):
        X_train = []
        y_train = []
        X_test = []
        y_test = []

        for j in range(k):
            if i == j:
                X_test = folds_X[j]
                y_test = folds_y[j]
            else:
                X_train += folds_X[j]
                y_train += folds_y[j]

        X_train = np.array(X_train)
        y_train = np.array(y_train)
        X_test = np.array(X_test)
        y_test = np.array(y_test)

        svm = SVM(Kernel.gaussian(0.3), 0.7)
        svm.train(X_train, y_train)

        f1s.append(vld.f1_measure(X_test, y_test, svm))

        # print(vld.confusion_matrix(X_test, y_test, svm))

    print("f1: " + str(sum(f1s) / len(f1s)))

    # y_guessed = [svm.predict(x) for x in X_test]
    # vis.plot(X, y, X_test, y_guessed, svm, 50)
