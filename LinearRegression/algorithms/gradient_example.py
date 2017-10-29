import numpy as np
from matplotlib import pyplot as plt

class LinRegression:
    '''Linear Regression.'''

    def __init__(self):
        self.about = "Linear Regression"
        self.W = []  # model's weights
        self.fscaling = False  # is feature scaling used

    def cost(self, y_real, y_pred):
        # cost function for gradient descent algorithm
        return np.sum((y_pred - y_real) ** 2) / (len(y_real))

    def gradient_descent_step(self, learning_rate, dy, m, n, X_tr):
        # one gradient descent step
        s = (np.dot(dy.T, X_tr)).reshape(n, 1)
        dW = 2 * (learning_rate * s / m).reshape(n, 1)
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
            kweigths=1,
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
        cost0 = self.cost(y, y_pred)
        y = y.reshape(m, 1)
        k = 0

        ########## Gradient descent's steps #########
        while True:
            dy = y_pred - y
            W_tmp = self.W
            self.W = self.gradient_descent_step(learning_rate, dy, m, n, X)
            y_pred = np.dot(X, self.W)
            cost1 = self.cost(y, y_pred)
            k += 1
            if (cost1 > cost0):
                self.W = W_tmp
                break

            if ((cost0 - cost1) < e) or (k == nsteps):
                break

            cost0 = cost1
        #############################################
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
best_steps = 0
best_error = 1.0e+30

""" Just example of calculations """
for rate in np.linspace(0.99, 0.999, 10, dtype=float):
    for weight_low in np.linspace(-1000, -500, 250, dtype=int):
        for weight_high in np.linspace(500, 1000, 250, dtype=int):
            print("rate:", rate, "; weight_low:", weight_low, "; weight_high:", weight_high)
            for steps in range(2500, 3500):
                lr = LinRegression()
                lr.fit(X, y, learning_rate=rate, random_state=0, weight_low=weight_low, weight_high=weight_high, nsteps=steps)
                xx = [i for i in range(X.shape[0])]
                y1 = lr.predict(X)
                error = mean_squared_error(y, y1)
                if error < best_error:
                    best_rate = rate
                    best_wl = weight_low
                    best_wh = weight_high
                    best_steps = steps
                    best_error = error


X = np.hstack((rooms, areas))
y = np.array(prices)
lr = LinRegression()
lr.fit(X, y, learning_rate = best_rate, random_state = 0, weight_low = best_wl, weight_high = best_wh, nsteps=best_steps)
xx = [i for i in range(X.shape[0])]
y1 = lr.predict(X)
print('MSE1 (My LR model):', best_error)
f=0
t=40
plt.plot(xx[f:t], y[f:t], color='r', linewidth=4, label='y')
plt.plot(xx[f:t], y1[f:t], color='b', linewidth=2, label='predicted y')
plt.ylabel('Target label')
plt.xlabel('Line number in dataset')
plt.legend(loc=4)
plt.show()