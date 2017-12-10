from svm import *
from kernel import Kernel
from reader import read_training_set
import shared.Visualizer as vis
import shared.Validator as vld

if __name__ == "__main__":
    num_samples = 100
    num_features = 2
    grid_size = 50

    # samples = np.matrix(np.random.normal(size=num_samples * num_features)
    #                     .reshape(num_samples, num_features))
    # labels = 2 * (samples.sum(axis=1) > 0) - 1.0
    # svm = SVM(Kernel.linear(), 0.1)

    X, y = read_training_set(shuf=True)
    svm = SVM(Kernel.gaussian(0.12), 0.1)
    svm.train(X, y)

    print(vld.f1_measure(X, y, svm))

    vis.plot(X, y, svm, grid_size)
