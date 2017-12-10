from svm import *
from kernel import Kernel
import shared.Visualizer as vis
import shared.Validator as vld
import shared.Reader as reader

if __name__ == "__main__":
    X, y = reader.read_training_set(shuf=True)
    svm = SVM(Kernel.gaussian(0.12), 0.1)
    svm.train(X, y)

    print(vld.f1_measure(X, y, svm))

    vis.plot(X, y, svm, 50)
