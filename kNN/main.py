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
from kNN.Validator import Validator
from kNN.Visualizer import Visualizer

MIN_K = 4
MAX_K = 10
MIN_FOLDS = 2
MAX_FOLDS = 15


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

    try: os.mkdir("results")
    except: pass

    """
    Iterate over: Transfomers, Folds, k, metrics and kernels.
    """

    avgs_for_global = [0 for i in range(0, len(transformers))]
    counts_for_global = [0 for i in range(0, len(transformers))]

    with open("results/avg.txt", "w") as avg_global_file:

        for transformer in transformers:
            print("Transformer: ", str(transformer))
            points = transformer.transform(data)

            try: os.mkdir("results/" + str(transformer))
            except: pass

            with open("results/" + str(transformer) + "/avg.txt", "w") as avg_transform_file:

                avgs_for_folds = [0 for i in range(MIN_FOLDS, MAX_FOLDS)]
                counts_for_folds = [0 for i in range(MIN_FOLDS, MAX_FOLDS)]

                # for each n-fold
                for folds in range(MIN_FOLDS, MAX_FOLDS):

                    with open("results/" + str(transformer) + "/" + str(folds) + "-folds.txt", "w") as file:

                        shuffle(points)

                        avgs_for_k = [0 for i in range(MIN_K, MAX_K)]
                        counts_for_k = [0 for i in range(MIN_K, MAX_K)]

                        print("Folds: ", str(folds))

                        test_size = len(points) // folds
                        for fold_number in range(folds):
                            test_set = [point for point in points if
                                        fold_number * test_size <= points.index(point) < (fold_number + 1) * test_size]
                            train_set = [point for point in points if point not in test_set]

                        # for each k (for kNN)
                        for k in range(MIN_K, MAX_K):
                            print("k: ", str(k))
                            file.write("For K = " + str(k) + ":\n\n")
                            max_f1 = -1
                            avg_f1_for_k = 0
                            count_for_k = 0

                            # for each metric
                            for metric in metrics:

                                # for each kernel
                                for kernel in kernels:
                                    classifier = SimpleKNNClassifier(k, kernel, metric)
                                    classifier.train(train_set)
                                    f1 = Validator.f1_measure(classifier, test_set)

                                    avg_f1_for_k += f1
                                    count_for_k += 1

                                    avgs_for_k[k - MIN_K] += f1
                                    counts_for_k[k - MIN_K] += 1

                                    avgs_for_folds[folds - MIN_FOLDS] += f1
                                    counts_for_folds[folds - MIN_FOLDS] += 1

                                    avgs_for_global[transformers.index(transformer)] += f1
                                    counts_for_global[transformers.index(transformer)] += 1

                                    if f1 >= max_f1:
                                        max_f1 = f1
                                        best_k = k
                                        best_folds = folds
                                        best_metric = str(metric)
                                        best_kernel = str(kernel)

                                    if max_f1 == 1.0:
                                        file.write("\tBest F1: " + str(max_f1) + "\n")
                                        file.write("\t" + str(metric) + ", " + str(kernel) + "\n\n")

                            if max_f1 != 1.0:
                                file.write("\tBest F1: " + str(max_f1) + "\n")
                                file.write("\t" + str(best_metric) + ", " + str(best_kernel) + "\n\n")

                        # append with for-each k stats
                        file.write("\n")
                        for k in range(MIN_K, MAX_K):
                            file.write("Average F for k = " + str(k) + ": " + str(avgs_for_k[k - MIN_K] / counts_for_k[k - MIN_K]) + "\n")
                            file.flush()

                        max_avg_for_k = -1
                        for k in range(MIN_K, MAX_K):
                            value = avgs_for_k[k - MIN_K] / counts_for_k[k - MIN_K]
                            if value > max_avg_for_k:
                                max_avg_for_k = value
                                max_k = k
                        file.write("\nMax F1 for k: " + str(max_avg_for_k) + ", k = " + str(max_k))
                        file.flush()

                # print stats for transformer
                for folds in range(MIN_FOLDS, MAX_FOLDS):
                    avg_transform_file.write("Average F for folds = " + str(folds) + ": " + str(avgs_for_folds[folds - MIN_FOLDS] / counts_for_folds[folds - MIN_FOLDS]) + "\n")
                    avg_transform_file.flush()

                max_avg_for_folds = -1
                for folds in range(MIN_FOLDS, MAX_FOLDS):
                    value = avgs_for_folds[folds - MIN_FOLDS] / counts_for_folds[folds - MIN_FOLDS]
                    if value > max_avg_for_folds:
                        max_avg_for_folds = value
                        max_folds = folds
                avg_transform_file.write("\nMax F1 for folds: " + str(max_avg_for_folds) + ", folds = " + str(max_folds))
                avg_transform_file.flush()

        # print status for global
        for transformer in transformers:
            avg_global_file.write("Average F for transformer = " + str(transformer) + ": " + str(avgs_for_global[transformers.index(transformer)] / counts_for_global[transformers.index(transformer)]) + "\n")
            avg_global_file.flush()

        max_avg_for_folds = -1
        for transformer in transformers:
            value = avgs_for_global[transformers.index(transformer)] / counts_for_global[transformers.index(transformer)]
            if value > max_avg_for_folds:
                max_avg_for_folds = value
                max_transformer = transformer
        avg_global_file.write("\nMax F1: " + str(max_avg_for_folds) + ", transformer = " + str(max_transformer))
        avg_global_file.flush()
