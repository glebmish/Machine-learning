from LinearRegression.reader import read_training_set
from LinearRegression.Visualizer import Visualizer
from LinearRegression.regression import LinRegression

def func(x, y):
    return x * y

if __name__ == '__main__':
    flats = read_training_set()

    regression = LinRegression()
    regression.fit(flats)
    print(regression.predict(flats[0]))

    vis = Visualizer()
    vis.visualize(flats, func)
