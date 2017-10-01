from kNN.KNNClassifier import KNNClassifier

class Validator(object):
    @staticmethod
    def correct_percentage(classifier, test_set):
        assert isinstance(classifier, KNNClassifier)

        guessed_cls = sum([1 for point in test_set if classifier.classify(point) == point.cls])
        return guessed_cls / len(test_set)

    @staticmethod
    def f1_measure(classifier, test_set):
        assert isinstance(classifier, KNNClassifier)

        guessed_cls = [classifier.classify(point) for point in test_set]
        correct_cls = [point.cls for point in test_set]

        # 0, 0 - tn; 0, 1 - fp; 1, 0 - fn; 1, 1 - tp
        counts = [[0, 0], [0, 0]]
        for cc, gc in zip(correct_cls, guessed_cls):
            counts[cc][gc] += 1

        recall = counts[1][1] / (counts[1][1] + counts[1][0])
        precision = counts[1][1] / (counts[1][1] + counts[0][1])

        f1_measure = 2 * precision * recall / (precision + recall)
        return f1_measure