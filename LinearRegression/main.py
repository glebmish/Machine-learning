from LinearRegression.reader import read_training_set
from LinearRegression.Visualizer import Visualizer
from LinearRegression.regression import LinRegression
from LinearRegression.methods import gradient, genetic
from LinearRegression.Flat import Flat

import numpy as np

if __name__ == '__main__':
    flats = read_training_set()

    gradient = gradient.Gradient()
    genetic = genetic.Genetic()

    regression = LinRegression(fit_method=genetic.fit)
    regression.fit(flats)

    print("Weights: {}".format(regression.W))
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
