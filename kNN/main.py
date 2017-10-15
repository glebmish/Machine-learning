from random import shuffle
from numpy import linspace

from kNN.kernels import Triangular
from kNN.kernels import Quartic
from kNN.kernels import Epanechnikov
from kNN.kernels import Gaussian
from kNN.kernels import Sigmoid
from kNN.metrics import Minkovsky
from kNN.reader import *
from kNN.SimpleKNNClassifier import *
from kNN.KNNClassifier import *
from kNN.Validator import Validator
from kNN.Visualizer import Visualizer

"""
if __name__ == '__main__':
    objects = read_training_set()
    shuffle(objects)

    train_set_len = len(objects) // 10 * 9

    train_set = objects[:train_set_len]
    test_set = objects[train_set_len:]

    k = 3
    classifier = KNNClassifier(k, class_resolver=resolve_class_kernel_function)

    classifier.train(train_set)

    metric = Validator.f1_measure(classifier, test_set)
    print("f1-measure={:.4f} for {}NN classifier".format(metric, k))

    classified_set = [Point(point.x, point.y, classifier.classify(point)) for point in test_set]

    visualizer = Visualizer()
    visualizer.visualize(train_set, test_set, classified_set)
"""

if __name__ == '__main__':
    points = read_training_set()
    shuffle(points)
    train_set_len = len(points) // 10 * 9

    train_set = points[:train_set_len]
    test_set = points[train_set_len:]

    max_f1 = -1

    for k in range(2, 31):
        for n in linspace(0.025, 3, 120):
            metric = Minkovsky.Minkovsky(n, 1)

            kernel = Triangular.Triangular(metric)
            classifier = SimpleKNNClassifier(k, kernel, metric)
            classifier.train(train_set)
            f1 = Validator.f1_measure(classifier, test_set)
            print("f1-measure={:.4f} for {}NN classifier, Triangular, {}-metric".format(f1, k, n))
            if f1 > max_f1:
                max_f1 = f1
                best_n = n
                best_k = k

            kernel = Quartic.Quartic(metric)
            classifier = SimpleKNNClassifier(k, kernel, metric)
            classifier.train(train_set)
            f1 = Validator.f1_measure(classifier, test_set)
            print("f1-measure={:.4f} for {}NN classifier, Quartic, {}-metric".format(f1, k, n))
            if f1 > max_f1:
                max_f1 = f1
                best_n = n
                best_k = k

            kernel = Epanechnikov.Epanechnikov(metric)
            classifier = SimpleKNNClassifier(k, kernel, metric)
            classifier.train(train_set)
            f1 = Validator.f1_measure(classifier, test_set)
            print("f1-measure={:.4f} for {}NN classifier, Epanechnikov, {}-metric".format(f1, k, n))
            if f1 > max_f1:
                max_f1 = f1
                best_n = n
                best_k = k

            kernel = Gaussian.Gaussian(metric)
            classifier = SimpleKNNClassifier(k, kernel, metric)
            classifier.train(train_set)
            f1 = Validator.f1_measure(classifier, test_set)
            print("f1-measure={:.4f} for {}NN classifier, Gaussian, {}-metric".format(f1, k, n))
            if f1 > max_f1:
                max_f1 = f1
                best_n = n
                best_k = k

            # kernel = Sigmoid.Sigmoid(metric)
            # classifier = SimpleKNNClassifier(k, kernel, metric)
            # classifier.train(train_set)
            # f1 = Validator.f1_measure(classifier, test_set)
            # print("f1-measure={:.4f} for {}NN classifier, Sigmoid, {}-metric".format(f1, k, n))
            # if f1 > max_f1:
            #     max_f1 = f1
            #     best_n = n
            #     best_k = k

    print(max_f1, best_n, best_k)
