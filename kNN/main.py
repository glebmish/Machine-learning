
from kNN.reader import *
from kNN.Point import Point
from kNN.KNNClassifier import KNNClassifier

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


if __name__ == '__main__':
    print("""
Choose mode:
    1. Guessing game
    """)

    type = int(input())
    if type == 1:
        guessing_game()
    else:
        print('Wrong mode')
