import numpy as np
from SupportVector.svm import *
from SupportVector.kernel import Kernel
from kNN.kNNClassifier import *
import shared.Reader as reader


def wilcoxon(X1, Y1, X2, Y2):
    diffs = [y1 - y2 for y1, y2 in zip(Y1, Y2)]
    diffs = [y for y in diffs if y != 0]

    print(diffs)


if __name__ == "__main__":
    X, y = reader.read_training_set(shuf=True)
    svm = SVM(Kernel.gaussian(0.12), 0.1)
    svm.train(X, y)
    guessed_label = [clf.predict(np.array(sample).reshape(1, 2)) for sample in X]

    knn = KNNClassifier()
    knn.train(X, y)
    guessed_label = [clf.predict(np.array(sample).reshape(1, 2)) for sample in X]

    wilcoxon()

