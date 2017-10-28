import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


class Visualizer(object):

    def visualize(self, train_set, function):
        fig = plt.figure()
        ax = fig.gca(projection='3d')

        self.__draw_points(ax, train_set, 'o', 'train')
        self.__draw_line(ax, train_set, function, 'function')

        plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.3), ncol=3)
        plt.tight_layout(pad=5)

        ax.set_xlabel('Area')
        ax.set_ylabel('Rooms')
        ax.set_zlabel('Price')

        plt.show()

    @classmethod
    def __draw_points(cls, ax, point_set, point_type, set_type, marker_size=8):
        xs, ys, zs = cls.__get_x_y_z(point_set)
        ax.scatter(xs, ys, zs, c='blue', marker=point_type)

    @classmethod
    def __draw_line(cls, ax, point_set, function, set_type, marker_size=8):
        xs, ys, zs = cls.__get_x_y_function(point_set, function)
        ax.plot(xs, ys, zs, c='red')

    @staticmethod
    def __get_x_y_z(point_set):
        x = [p.area for p in point_set]
        y = [p.rooms for p in point_set]
        z = [p.price for p in point_set]

        return x, y, z

    @staticmethod # Just an example function to render
    def __get_x_y_function(point_set, function):
        theta = np.linspace(-4 * np.pi, 4 * np.pi, 100)
        z = np.linspace(100000, 700000, 100)
        x = 1000 * np.sin(theta) + 1000
        y = 1 * np.cos(theta) + 2

        return x, y, z
