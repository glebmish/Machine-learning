from random import shuffle

from kNN.reader import *
from kNN.Point import Point
from kNN.KNNClassifier import KNNClassifier
from kNN.Validator import Validator

def guessing_game():
    print('Guessing game')

    objects = read_training_set()

    classifier = KNNClassifier(1)
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

    correct_percentage = Validator.correct_percentage(classifier, test_set)
    print("{0}NN classifier correctly classified {1:.2f}% of test set".format(k, correct_percentage * 100))


if __name__ == '__main__':
#     print("""
# Choose mode:
#     1. Guessing game
#     2. Validation 90% train 10% test
#     """)
#
#     type = int(input())

    type = 2

    if type == 1:
        guessing_game()
    elif type == 2:
        validation_90_10()
    else:
        print('Wrong mode')
