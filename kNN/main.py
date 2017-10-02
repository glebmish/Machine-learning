from random import shuffle

from kNN.reader import *
from kNN.Point import Point
from kNN.KNNClassifier import KNNClassifier
from kNN.Validator import Validator
from kNN.Visualizer import Visualizer


def guessing_game():
    print('Guessing game')

    objects = read_training_set()

    k = 1
    classifier = KNNClassifier(k)

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


def validation_90_10():
    objects = read_training_set()
    shuffle(objects)

    train_set_len = len(objects) // 10 * 9

    train_set = objects[:train_set_len]
    test_set = objects[train_set_len:]

    k = 1
    classifier = KNNClassifier(k)

    classifier.train(train_set)

    metric = Validator.f1_measure(classifier, test_set)
    print("f1-measure={:.4f} for {}NN classifier".format(metric, k))


def visualization_90_10():
    objects = read_training_set()
    shuffle(objects)

    train_set_len = len(objects) // 10 * 9

    train_set = objects[:train_set_len]
    test_set = objects[train_set_len:]

    k = 3
    classifier = KNNClassifier(k)

    classifier.train(train_set)

    for point in test_set:
        point.cls = classifier.classify(point)

    visualizer = Visualizer()
    visualizer.visualize(train_set, test_set)


if __name__ == '__main__':
    type = 3

    if type == 1:
        guessing_game()
    elif type == 2:
        validation_90_10()
    elif type == 3:
        visualization_90_10()
    else:
        print('Wrong mode')
