from random import shuffle

from kNN.reader import *
from kNN.Point import Point
from kNN.KNNClassifier import KNNClassifier
from kNN.Validator import Validator
from kNN.Visualizer import Visualizer
from kNN.KDTree.KNNKDClassifier import KNNKDClassifier


def guessing_game(method):
    print('Guessing game')

    objects = read_training_set()

    if method == 1:
        print("Using list")
        classifier = KNNClassifier(1)
    else:
        print("Using KDTree")
        classifier = KNNKDClassifier(1)

    classifier.train(objects)

    while True:
        try:
            x = float(input('Type x value: '))
            y = float(input('Type y value: '))
        except ValueError as e:
            print('Wrong value')
            continue

        point = Point(x, y)
        point.cls = classifier.classify(point)
        print(point)
        print()


def validation_90_10(method):
    objects = read_training_set()
    shuffle(objects)

    train_set_len = len(objects) // 10 * 9

    train_set = objects[:train_set_len]
    test_set = objects[train_set_len:]

    k = 1
    if method == 1:
        print("Using list")
        classifier = KNNClassifier(1)
    else:
        print("Using KDTree")
        classifier = KNNKDClassifier(1)

    classifier.train(train_set)

    metric = Validator.f1_measure(classifier, test_set)
    print("f1-measure={:.4f} for {}NN classifier".format(metric, k))


def visualization_90_10(method):
    objects = read_training_set()
    shuffle(objects)

    train_set_len = len(objects) // 10 * 9

    train_set = objects[:train_set_len]
    test_set = objects[train_set_len:]

    k = 3
    if method == 1:
        print("Using list")
        classifier = KNNClassifier(1)
    else:
        print("Using KDTree")
        classifier = KNNKDClassifier(1)

    classifier.train(train_set)

    for point in train_set:
        point = classifier.classify(point)

    visualizer = Visualizer()
    visualizer.visualize(train_set, test_set)


if __name__ == '__main__':
    type = 2
    method = 2

    if type == 1:
        guessing_game(method)
    elif type == 2:
        validation_90_10(method)
    elif type == 3:
        visualization_90_10(method)
    else:
        print('Wrong mode')
