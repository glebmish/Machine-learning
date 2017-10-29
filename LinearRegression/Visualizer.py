import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


class Visualizer(object):

    def visualize(self, train_set, weights):
        fig = plt.figure()
        ax = fig.gca(projection='3d')

        self.__draw_points(ax, train_set, 'o')
        self.__draw_function(ax, weights)

        plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.3), ncol=3)
        plt.tight_layout(pad=5)

        ax.set_xlabel('Area')
        ax.set_ylabel('Rooms')
        ax.set_zlabel('Price')

        plt.show()

    @classmethod
    def __draw_points(cls, ax, point_set, point_type, marker_size=8):
        X = [x.area for x in point_set]
        Y = [x.rooms for x in point_set]
        Z = [x.price for x in point_set]

        ax.scatter(X, Y, Z, c='blue', marker=point_type)

    @staticmethod
    def __draw_function(ax, weights, marker_size=8):
        X = np.linspace(1, 4000, 20)
        Y = np.linspace(1, 6, 20)
        Z = X * weights[0] + Y * weights[1]

        print(Z)

        ax.plot(X, Y, Z, c='red')
