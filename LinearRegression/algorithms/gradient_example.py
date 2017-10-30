import numpy as np
from matplotlib import pyplot as plt

class LinRegression:
    '''Linear Regression.'''

    def __init__(self):
        self.about = "Linear Regression"
        self.W = []  # model's weights
        self.fscaling = False  # is feature scaling used

    def error(self, y_real, y_pred):
        # error function for gradient descent algorithm
        return np.sum((y_pred - y_real) ** 2) / (len(y_real))

    def error_derivative(self, dy, X_tr, n, m):
        return (np.dot(dy.T, X_tr)).reshape(n, 1) * 2 / m

    def gradient_descent_step(self, learning_rate, dy, m, n, X_tr):
        # one gradient descent step
        err_derivative = self.error_derivative(dy, X_tr, n, m)
        dW = (learning_rate * err_derivative).reshape(n, 1)
        return self.W - dW

    def normalize(self, X):
        # normilize X table
        for j in range(X.shape[1]):
            X[:, j] = X[:, j] / np.max(X[:, j])
        return X

    def fit(self,
            X,
            y,
            learning_rate=0.99,
            nsteps=3000,
            e=0.000000001,
            weight_low=0,
            weight_high=1,
            fscaling=False,
            random_state=0
            ):
        # train our Linear Regression model

        np.random.seed(random_state)
        X = X.astype(float)

        # Normilize process
        if fscaling:
            X = self.normalize(X)
            self.fscaling = True

        m = X.shape[0]
        # add one's column to X
        X = np.hstack((np.ones(m).reshape(m, 1), X))
        n = X.shape[1]

        # Weights: random initialization
        self.W = np.random.randint(low=weight_low, high=weight_high, size=(n, 1))

        y_pred = np.dot(X, self.W)
        error_prev = self.error(y, y_pred)
        y = y.reshape(m, 1)
        k = 0

        # Gradient descent
        while True:
            dy = y_pred - y
            W_tmp = self.W
            self.W = self.gradient_descent_step(learning_rate, dy, m, n, X)
            y_pred = np.dot(X, self.W)
            error = self.error(y, y_pred)
            k += 1
            if error > error_prev:
                self.W = W_tmp
                break

            if error_prev - error < e or k == nsteps:
                break

            error_prev = error

        return self.W  # return model's weights

    def predict(self, X):
        m = X.shape[0]
        if not self.fscaling:
            return np.dot(
                np.hstack(
                    (np.ones(m).reshape(m, 1),
                     X.astype(float))
                ),
                self.W)
        else:
            return np.dot(
                np.hstack(
                    (np.ones(m).reshape(m, 1),
                     self.normalize(X.astype(float)))
                ),
                self.W)


def mean_error(y1, y2):
    Y1 = np.array(y1)
    Y2 = np.array(y2)
    return np.sqrt(np.sum((Y1 - Y2) ** 2) / (len(Y1)))


from LinearRegression.reader import read_training_set
from sklearn.metrics import mean_squared_error

flats = read_training_set()

rooms = [[flat.rooms] for flat in flats]
areas = [[flat.area] for flat in flats]
prices = [flat.price for flat in flats]

X = np.hstack((rooms, areas))
y = np.array(prices)

best_rate = 0
best_wl = 0
best_wh = 0
best_error = 1.0e+30
"""
for rate in np.linspace(0.996, 1, 5, dtype=float):
    for weight_low in np.linspace(-400, -600, 200, dtype=int):
        if weight_low % 40 == 0: print(rate, weight_low)
        for weight_high in np.linspace(400, 600, 200, dtype=int):
            lr = LinRegression()
            lr.fit(X, y, learning_rate=0.997, random_state=0, weight_low=weight_low, weight_high=weight_high, nsteps=3000)
            xx = [i for i in range(X.shape[0])]
            y1 = lr.predict(X)
            error = mean_squared_error(y, y1)
            if error < best_error:
                best_rate = rate
                best_wl = weight_low
                best_wh = weight_high
                best_error = error
"""

from sklearn.linear_model import LinearRegression

print(best_rate, best_wl, best_wh)

X = np.hstack((rooms, areas))
y = np.array(prices)
lr1 = LinRegression()

lr2 = LinearRegression()
lr2.fit(X,y)

# 0.996 -464 400
lr1.fit(X, y, learning_rate=0.996, random_state=0, weight_low=-464, weight_high=400, nsteps=3000, fscaling=False)
# lr.fit(X, y, learning_rate=best_rate, random_state=0, weight_low=best_wl, weight_high=best_wh, nsteps=3000, fscaling=False)

xx = [i for i in range(X.shape[0])]
y1 = lr1.predict(X)
y2 = lr2.predict(X)
print('MSE1 (My LR model):', mean_squared_error(y, y1))
print('Error (My LR model):', mean_error(y, y1))
print('MSE1 (My LR model):', mean_squared_error(y, y2))
print('Error (My LR model):', mean_error(y, y2))

f = 0
t = len(y)
plt.plot(xx[f:t], y[f:t], color='r', linewidth=4, label='y')
plt.plot(xx[f:t], y1[f:t], color='b', linewidth=2, label='predicted')
plt.plot(xx[f:t], y2[f:t], color='g', linewidth=2, label='predicted y by sklearn')
plt.ylabel('Target label')
plt.xlabel('Line number in dataset')
plt.legend(loc=4)
plt.show()