from Bayes.document import Document
from Bayes.Bayes_classifier import Bayes
from Bayes.reader import read_buckets
from Bayes.CrossValidation import validate
from Bayes.CrossValidation import count_f


if __name__ == "__main__":
    documents_buckets = read_buckets()
    debug_print = False
    cross_validation = False

    if cross_validation:
        f_measures = validate(Bayes, documents_buckets, debug_print)
        for i in range(len(f_measures)):
            print(i, " - ", f_measures[i])
    else:
        train_set = sum(documents_buckets[:-2], [])
        test_set = documents_buckets[-1]
        bayes = Bayes()
        bayes.train(train_set)
        f_measure = count_f(bayes, test_set, True)
        print(f_measure)