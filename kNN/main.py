from random import shuffle

from kNN.reader import *
from kNN.KNNClassifier import *
from kNN.Validator import Validator
from kNN.Visualizer import Visualizer


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
