from LinearRegression.reader import read_training_set
from LinearRegression.Visualizer import Visualizer
from LinearRegression.regression import LinRegression
from LinearRegression.algorithms import gradient

if __name__ == '__main__':
    flats = read_training_set()

    gradient = gradient.Gradient(weight_low=1, weight_high=2)

    regression = LinRegression(fit_method=gradient.fit)
    regression.fit(flats)

    vis = Visualizer()
    vis.visualize(flats, regression)

    input('Press Enter to exit...')
