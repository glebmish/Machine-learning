
from kNN.reader import *
from kNN.KNNClassifier import KNNClassifier

if __name__ == '__main__':
    objects = read_training_set()

    classifier = KNNClassifier(1)
    classifier.train(objects)

    while True:
        try:
            x = float(input('Type x value: '))
            y = float(input('Type y value: '))
        except ValueError as e:
            continue

        point = Object(x, y)
        point.cls = classifier.classify(point)
        print(point)
        print()
