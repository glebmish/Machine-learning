from LinearRegression.reader import read_training_set
from LinearRegression.Visualizer import Visualizer
from LinearRegression.regression import LinRegression
from LinearRegression.methods import gradient, genetic
import LinearRegression.algorithms.genetic as another_genetic
from LinearRegression.Flat import Flat

import numpy as np

# точное решение линейной регрессии, псевдообратная матрица, метод наименьших квадратов
if __name__ == '__main__':
    flats = read_training_set()

    gradient = gradient.Gradient()
    genetic = genetic.Genetic()

    another_genetic = another_genetic.Genetic(nsteps=2000, mutation_rate=0.85, tournament_size=10, weight_low=-10000, weight_high=30000)

    regression = LinRegression(regression_method=another_genetic)
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
