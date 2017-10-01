from kNN.KNNClassifier import KNNClassifier

class Validator(object):
    @staticmethod
    def correct_percentage(classifier, test_set):
        assert isinstance(classifier, KNNClassifier)
        guessed_cls = sum([1 for point in test_set if classifier.classify(point) == point.cls])
        return guessed_cls / len(test_set)