import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import itertools


def plot(X, y, X_test, y_guessed, clf, grid_size=50):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, grid_size),
                         np.linspace(y_min, y_max, grid_size),
                         indexing='ij')
    flatten = lambda m: np.array(m).reshape(-1,)

    result = []
    for (i, j) in itertools.product(range(grid_size), range(grid_size)):
        point = np.array([xx[i, j], yy[i, j]]).reshape(1, 2)
        result.append(clf.predict(point))

    Z = np.array(result).reshape(xx.shape)

    plt.contourf(xx, yy, Z,
                 cmap=cm.Paired,
                 levels=[-0.001, 0.001],
                 extend='both',
                 alpha=0.8)
    plt.scatter(flatten(X[:, 0]), flatten(X[:, 1]),
                c=flatten(y), cmap=cm.Paired)
    plt.scatter(flatten(X_test[:, 0]), flatten(X_test[:, 1]),
                c=flatten(y_guessed), cmap=cm.Paired, s=4)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.show()