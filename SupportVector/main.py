import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import itertools

from svm import *
from kernel import Kernel
from reader import read_training_set


def plot(svm, X, y, grid_size):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, grid_size),
                         np.linspace(y_min, y_max, grid_size),
                         indexing='ij')
    flatten = lambda m: np.array(m).reshape(-1,)

    result = []
    for (i, j) in itertools.product(range(grid_size), range(grid_size)):
        point = np.array([xx[i, j], yy[i, j]]).reshape(1, 2)
        result.append(svm.predict(point))

    Z = np.array(result).reshape(xx.shape)

    plt.contourf(xx, yy, Z,
                 cmap=cm.Paired,
                 levels=[-0.001, 0.001],
                 extend='both',
                 alpha=0.8)
    plt.scatter(flatten(X[:, 0]), flatten(X[:, 1]),
                c=flatten(y), cmap=cm.Paired)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.show()


if __name__ == "__main__":
    num_samples = 100
    num_features = 2
    grid_size = 50

    # samples = np.matrix(np.random.normal(size=num_samples * num_features)
    #                     .reshape(num_samples, num_features))
    # labels = 2 * (samples.sum(axis=1) > 0) - 1.0
    # svm = SVM(Kernel.linear(), 0.1)

    samples, labels = read_training_set(shuf=True)
    svm = SVM(Kernel.gaussian(0.12), 0.1)
    svm.train(samples, labels)

    plot(svm, samples, labels, grid_size)
