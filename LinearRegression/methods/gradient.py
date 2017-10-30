import numpy as np

from LinearRegression.methods.base import Base


class Gradient(Base):

    def __init__(self):
        super().__init__()

    def fit(self, data_X, data_Y, alpha=1.0e-2, steps=2000, normalize=True):
        X = np.zeros(data_X.shape)

        self.normalize = normalize

        if self.normalize:
            X[:, 0] = data_X[:, 0] / np.max(data_X[:, 0])
            X[:, 1] = data_X[:, 1] / np.max(data_X[:, 1])
            Y = data_Y / np.max(data_Y)
            self.y_max = np.max(data_Y)
        else:
            X = data_X
            Y = data_Y

        W = np.array([0, 0, 0], dtype=float)
        for step in range(steps):
            Y_predited = W[0] + W[1] * X[:, 0] + W[2] * X[:, 1]
            delta = Y - Y_predited
            for i in range(X.shape[0]):
                W = W + alpha * np.array([delta[i], delta[i] * X[i, 0], delta[i] * X[i, 1]])
        self.W = W


if __name__ == "__main__":
    data = np.loadtxt("../prices.txt", skiprows=1, delimiter=',').astype(int)
    X, Y = data[..., 0:2], data[..., 2]
    regression = Gradient()
    regression.fit(X, Y)
    Y_predicted = regression.predict(X)
    print(regression.mean_deviation(Y, Y_predicted))
