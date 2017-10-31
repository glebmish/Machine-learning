from LinearRegression.reader import read_training_set
from LinearRegression.Visualizer import Visualizer
from LinearRegression.regression import LinRegression
from LinearRegression.methods import gradient, genetic
import LinearRegression.algorithms.genetic as genetic1
from LinearRegression.Flat import Flat

import numpy as np

# точное решение линейной регрессии, псевдообратная матрица, метод наименьших квадратов
if __name__ == '__main__':
    flats = read_training_set()

    gradient = gradient.Gradient()
    genetic = genetic.Genetic()

    genetic1 = genetic1.Genetic(nsteps=1500, mutation_rate=0.70)

    regression = LinRegression(regression_method=genetic1)
    regression.fit(flats)

    y_real = np.array([flat.price for flat in flats])
    y_pred = regression.predict(flats)
    error = regression.mean_deviation(y_real, y_pred)

    print("Weights: {}".format(regression.W))
    print("Error: {}".format(error))

    vis = Visualizer()
    vis.visualize(flats, regression)

    while True:
        try:
            area = input("type area ('q' to exit): ")
            if area == 'q':
                break
            else:
                area = float(area)

            rooms = input("type number of rooms ('q' to exit): ")
            if rooms == 'q':
                break
            else:
                rooms = int(rooms)
        except ValueError:
            print("Wrong format\n")
            continue

        print("Flat price = {}".format(int(regression.predict([Flat(area, rooms)])[0])))
