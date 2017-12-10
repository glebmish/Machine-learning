import matplotlib.pyplot as plt


class Visualizer(object):

    def visualize(self, train_set, test_set, classified_set):
        self.__draw_points(train_set, 'o', 'train')
        self.__draw_points(classified_set, '^', 'classified', marker_size=15)
        self.__draw_points(test_set, '^', 'test')

        plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.3), ncol=3)
        plt.tight_layout(pad=5)
        self.__show()

    @classmethod
    def __draw_points(cls, point_set, point_type, set_type, marker_size=8):
        pts_1 = [p for p in point_set if p.cls == 1]
        x_1, y_1 = cls.__get_x_y(pts_1)

        point_type_1 = 'b{}'.format(point_type)
        label_1 = '1 type {} set'.format(set_type)

        pts_0 = [p for p in point_set if p.cls == 0 or p.cls == -1]
        x_0, y_0 = cls.__get_x_y(pts_0)

        point_type_0 = 'y{}'.format(point_type)
        label_0 = '0 type {} set'.format(set_type)

        plt.plot(x_1, y_1, point_type_1, label=label_1, ms=marker_size)
        plt.plot(x_0, y_0, point_type_0, label=label_0, ms=marker_size)

    @staticmethod
    def __get_x_y(point_set):
        x = [p.x for p in point_set]
        y = [p.y for p in point_set]

        return x, y

    def __show(self):
        plt.show()
