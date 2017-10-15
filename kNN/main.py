from random import shuffle
from numpy import linspace

from kNN.kernels import Triangular
from kNN.kernels import Quartic
from kNN.kernels import Epanechnikov
from kNN.kernels import Gaussian
from kNN.kernels import Sigmoid
from kNN.metrics import Minkovsky
from kNN.transformers import Linear
from kNN.transformers import Tan
from kNN.transformers import Rad

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
    data = read_training_set()
    shuffle(data)

    visualized = False

    metrics = []
    for n in linspace(0.1, 3, 30):
        metrics.append(Minkovsky.Minkovsky(n))

    kernels = [Triangular.Triangular(),
               Quartic.Quartic(),
               Epanechnikov.Epanechnikov(),
               Gaussian.Gaussian(),
               Sigmoid.Sigmoid()]

    transformers = [Linear.Linear(),
                    Tan.Tan(),
                    Rad.Rad()]

    for transformer in transformers:
        print("Transformer: ", str(transformer))
        points = transformer.transform(data)
        with open("result-" + str(transformer) + ".txt", "w") as file:
            max_f1 = 0
            for folds in range(2, 16):
                print("Folds: ", str(folds))
                test_size = len(points) // folds
                for fold_number in range(folds):
                    test_set = [point for point in points if
                                fold_number * test_size <= points.index(point) < (fold_number + 1) * test_size]
                    train_set = [point for point in points if point not in test_set]

                for k in range(4, int(math.sqrt(len(train_set)))):
                    print("k: ", str(k))
                    for metric in metrics:
                        for kernel in kernels:
                            classifier = SimpleKNNClassifier(k, kernel, metric)
                            classifier.train(train_set)
                            f1 = Validator.f1_measure(classifier, test_set)
                            # print("f1-measure={:.4f} for {}NN classifier, {}, {} metric".format(f1, k, kernel, metric))
                            if f1 >= max_f1:
                                max_f1 = f1
                                best_k = k
                                best_folds = folds
                                best_metric = str(metric)
                                best_kernel = str(kernel)
                                best_transformer = str(transformer)
                                file.write("Best F1: " + str(max_f1) + "\n")
                                file.write("Best F1: " + str(max_f1) + "\n")
                                file.write(str(best_transformer) + " transform, " + str(best_folds) + " folds,\n")
                                file.write(str(best_k) + "-NN, " + str(best_metric) + ", " + str(best_kernel) + "\n\n")
                                file.flush()
