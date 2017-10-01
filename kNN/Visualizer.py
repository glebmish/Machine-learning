import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

class Visualizer(object):

    def visualize(self, train_point_set, test_point_set):
        self.__visualize_circles(train_point_set)
        self.__visualize_triangles(test_point_set)
        plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.3), ncol=2)
        plt.tight_layout(pad=5)
        self.__show()


    def __visualize_circles(self, point_set):
        blue_pts = [p for p in point_set if p.cls == 1]
        blue_x, blue_y = self.__get_x_y(blue_pts)

        red_pts = [p for p in point_set if p.cls == 0]
        red_x, red_y = self.__get_x_y(red_pts)

        plt.plot(blue_x, blue_y, 'bo', label='1 type train set')
        plt.plot(red_x, red_y, 'ro', label='0 type train set')

    def __visualize_triangles(self, point_set):
        blue_pts = [p for p in point_set if p.cls == 1]
        blue_x, blue_y = self.__get_x_y(blue_pts)

        red_pts = [p for p in point_set if p.cls == 0]
        red_x, red_y = self.__get_x_y(red_pts)

        plt.plot(blue_x, blue_y, 'b^', label='1 type test set')
        plt.plot(red_x, red_y, 'r^', label='0 type test set')

    @staticmethod
    def __get_x_y(point_set):
        x = [p.x for p in point_set]
        y = [p.y for p in point_set]

        return x, y

    def __show(self):
        plt.show()
