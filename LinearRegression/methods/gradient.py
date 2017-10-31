"""
    На каждом шаге, зная W, можем посчитать разницу.
    Y_predicted = WX
    Delta = Y - Y_predicted
    New W = W + alpha * Delta*X ( = derivative)
    alpha - шаг спуска
"""
import numpy as np
from LinearRegression.methods.base import Base


class Gradient(Base):

    def __init__(self, alpha=1.0e-2, steps=2000, normalize=True):
        super().__init__()
        self.alpha = alpha
        self.steps = steps
        self.normalize = normalize

    def fit(self, data_X, data_Y):
        X = np.ones([data_X.shape[0], data_X.shape[1] + 1])

        if self.normalize:
            X[:, 1] = data_X[:, 0] / np.max(data_X[:, 0])
            X[:, 2] = data_X[:, 1] / np.max(data_X[:, 1])
            Y = data_Y / np.max(data_Y)
            self.x_max = np.array([np.max(data_X[:, 0]), np.max(data_X[:, 1])])
            self.y_max = np.max(data_Y)
        else:
            X = data_X
            Y = data_Y

        W = np.array([0, 0, 0], dtype=float)
        for step in range(self.steps):
            Y_predited = np.dot(X, W)
            delta = Y - Y_predited
            W = W + self.alpha * np.dot(delta, X)

        self.W = W
        return self.W


if __name__ == "__main__":
    data = np.loadtxt("../prices.txt", skiprows=1, delimiter=',').astype(int)
    X, Y = data[..., 0:2], data[..., 2]
    regression = Gradient()
    regression.fit(X, Y)
    Y_predicted = regression.predict(X)
    print(regression.mean_deviation(Y, Y_predicted))

    while True:
        area = input("Area: ")
        rooms = input("Rooms: ")
        print(regression.predict_single(np.array([int(area), int(rooms)])))