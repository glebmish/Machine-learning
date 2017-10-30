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

    def __init__(self):
        super().__init__()

    def fit(self, data_X, data_Y, alpha=1.0e-2, steps=2000, normalize=True):
        X = np.ones([data_X.shape[0], data_X.shape[1] + 1])

        self.normalize = normalize

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
        for step in range(steps):
            # Y_predited = W[0] * X[:, 0] + W[1] * X[:, 1] + W[2] * X[:, 2]
            # delta = Y - Y_predicted
            # for i in range(X.shape[0]):
            #    W = W + alpha * np.array([delta[i]*X[i, 0], delta[i] * X[i, 1], delta[i] * X[i, 2]])
            Y_predited = np.dot(X, W)
            delta = Y - Y_predited
            W = W + alpha * np.dot(delta, X)
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