"""
    Есть n популяций (около 100)
    В каждой популяции создется m (около 50) векторов, забиваются случайными коэффициентами (-1 .. 1)
    Делается steps шагов
        Выбирается рандомная популяция
        Для каждого представителя 'a' в этой популяции:
            k = random(0.5 .. 1)
            Выбираются три рандомных вектора: b, c, d
            new_a = b + k * (c - d)
            if new_a better then a:
                a = new_a: смотрится по отклонению
"""


import numpy as np
from LinearRegression.methods.base import Base


class Genetic(Base):

    def __init__(self, n=100, m=50, steps=2000, normalize=True):
        super().__init__()

        self.n = n
        self.m = m
        self.steps = steps
        self.normalize = normalize

    def fit(self, data_X, data_Y):
        X = np.ones([data_X.shape[0], data_X.shape[1] + 1])

        if self.normalize:
            X[:, 1] = data_X[:, 0] / np.max(data_X[:, 0])
            X[:, 2] = data_X[:, 1] / np.max(data_X[:, 1])
            self.x_max = np.array([np.max(data_X[:, 0]), np.max(data_X[:, 1])])
            self.y_max = np.max(data_Y)
        else:
            X = data_X

        best_error = 10000000000

        np.random.seed(self.n * self.m)
        populations = np.random.rand(self.n, self.m, 3)
        for step in range(self.steps):
            if step % 1000 == 0: print(step)
            population = populations[np.random.randint(0, self.n)]
            for index in range(self.m):
                a = population[index]
                k = np.random.rand() / 2 + 0.5
                b = population[np.random.randint(0, self.m)]
                c = population[np.random.randint(0, self.m)]
                d = population[np.random.randint(0, self.m)]
                a_new = b + k * (c - d)

                if self.normalize:
                    Y_predicted_old = np.dot(X, a) * np.max(data_Y)
                    Y_predicted_new = np.dot(X, a_new) * np.max(data_Y)
                else:
                    Y_predicted_old = np.dot(X, a)
                    Y_predicted_new = np.dot(X, a_new)

                old_error = self.mean_deviation(data_Y, Y_predicted_old)
                new_error = self.mean_deviation(data_Y, Y_predicted_new)
                if new_error < old_error:
                    population[index] = a_new
                    if new_error < best_error:
                        best_error = new_error
                        self.W = a_new
        return self.W


if __name__ == "__main__":
    data = np.loadtxt("../prices.txt", skiprows=0, delimiter=',').astype(int)
    X, Y = data[..., 0:2], data[..., 2]
    regression = Genetic()
    regression.fit(X, Y)
    Y_predicted = regression.predict(X)
    print(regression.mean_deviation(Y, Y_predicted))

    while True:
        area = input("Area: ")
        rooms = input("Rooms: ")
        print(regression.predict_single(np.array([int(area), int(rooms)])))