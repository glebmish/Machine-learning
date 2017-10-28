from LinearRegression.reader import read_training_set
from LinearRegression.Visualizer import Visualizer


def func(x, y):
    return x * y


if __name__ == '__main__':
    points = read_training_set()
    vis = Visualizer()
    vis.visualize(points, func)
