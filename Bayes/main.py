from Bayes.Bayes_classifier import Bayes
from Bayes.reader import read_buckets
from Bayes.CrossValidation import validate
from Bayes.CrossValidation import count_f


if __name__ == "__main__":
    documents_buckets = read_buckets()
    debug_print = False
    cross_validation = True

    bayes = Bayes(include_subject=True, min_occurence=None)

    if cross_validation:
        f_measures = validate(bayes, documents_buckets, debug_print)

        average_f = sum(f_measures) / len(f_measures)
        print("Average f measure: ", average_f)
    else:
        train_set = sum(documents_buckets[:-2], [])
        test_set = documents_buckets[-1]

        bayes.train(train_set)
        f_measure = count_f(bayes, test_set, True)
        print(f_measure)