from LinearRegression.reader import read_training_set
from LinearRegression.Visualizer import Visualizer
from LinearRegression.regression import LinRegression
from LinearRegression.algorithms import gradient
from LinearRegression.Flat import Flat

if __name__ == '__main__':
    flats = read_training_set()

    gradient = gradient.Gradient(weight_low=1, weight_high=2)

    regression = LinRegression(fit_method=gradient.fit)
    regression.fit(flats)

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
