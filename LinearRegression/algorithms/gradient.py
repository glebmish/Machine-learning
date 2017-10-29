import numpy as np


class Gradient(object):

    def __init__(self, learning_rate=0.99,
                 nsteps=3000, e=0.0000000001,
                 weight_low=0, weight_high=1, random_seed=1):
        self.learning_rate = learning_rate
        self.nsteps = nsteps
        self.e = e
        self.weight_low = weight_low
        self.weight_high = weight_high
        self.random_seed = random_seed

    def fit(self, X, y):
        np.random.seed(self.random_seed)
        X = X.astype(float)

        m = X.shape[0]
        # add 1s column to also find w0 (absolute term of regression)
        X = np.hstack((np.ones(m).reshape(m, 1), X))
        n = X.shape[1]

        W = np.random.randint(low=self.weight_low, high=self.weight_high,
                              size=(n, 1))

        y_pred = np.dot(X, W)
        error0 = self.error(y, y_pred)

        y = y.reshape(m, 1)
        steps = 0

        while True:
            dy = y_pred - y
            W_tmp = W

            W = self.gradient_descent_step(W, dy, m, n, X)
            y_pred = np.dot(X, W)

            error1 = self.error(y, y_pred)
            steps += 1

            if error1 > error0:
                return W_tmp

            if (error0 - error1) < e or steps == self.nsteps:
                return W

            error0 = error1

    @staticmethod
    def error(y_real, y_pred):
        # mean squared error
        return np.sum((y_pred - y_real) ** 2) / len(y_real)

    def gradient_descent_step(self, W, dy, m, n, X):
        s = (np.dot(dy.T, X)).reshape(n, 1)
        dW = 2 * (s * self.learning_rate / m).reshape(n, 1)
        return W - dW
